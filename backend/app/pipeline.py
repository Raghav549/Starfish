from __future__ import annotations
from dataclasses import dataclass
import json,math
import cv2,numpy as np
from skimage.morphology import skeletonize,remove_small_objects

@dataclass
class FingerprintPipeline:
    block_size:int=16
    gabor_sigma:float=4.0
    min_distance:int=10
    border_margin:int=12
    min_component:int=24
    def _normalize(self,img):
        x=img.astype(np.float32); p1,p99=np.percentile(x,(1,99))
        return np.clip((x-p1)/max(p99-p1,1)*255,0,255).astype(np.uint8)
    def _quality(self,img):
        gx=cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)
        energy=cv2.GaussianBlur(gx*gx+gy*gy,(0,0),5)
        gxx=float(np.mean(gx*gx)); gyy=float(np.mean(gy*gy)); gxy=float(np.mean(gx*gy))
        coh=math.sqrt((gxx-gyy)**2+4*gxy*gxy)/(gxx+gyy+1e-6)
        return {"ridge_energy":float(np.mean(energy)),"coherence":float(coh)}
    def _segment(self,img):
        gx=cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)
        e=cv2.GaussianBlur(gx*gx+gy*gy,(0,0),7); mask=(e>max(float(np.percentile(e,45)),1e-4)).astype(np.uint8)
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)); mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k); mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
        n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8); clean=np.zeros_like(mask); lim=max(self.min_component,int(mask.size*.002))
        for i in range(1,n):
            if stats[i,cv2.CC_STAT_AREA]>=lim: clean[lab==i]=1
        return clean*255
    def _orientation_frequency(self,img):
        h,w=img.shape; bs=self.block_size
        gx=cv2.Sobel(img.astype(np.float32),cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(img.astype(np.float32),cv2.CV_32F,0,1,ksize=3)
        gxx=cv2.GaussianBlur(gx*gx,(0,0),3); gyy=cv2.GaussianBlur(gy*gy,(0,0),3); gxy=cv2.GaussianBlur(gx*gy,(0,0),3)
        local=.5*np.arctan2(2*gxy,gxx-gyy+1e-6)+math.pi/2; ori=cv2.GaussianBlur(local,(0,0),3); freq=np.zeros_like(ori)
        for y in range(0,h-bs+1,bs):
            for x in range(0,w-bs+1,bs):
                theta=float(np.median(local[y:y+bs,x:x+bs])); yy,xx=np.indices((bs,bs)); coord=xx*np.cos(theta)+yy*np.sin(theta)
                bins=np.arange(-bs,bs+1); prof=np.array([img[y:y+bs,x:x+bs][np.abs(coord-b)<.5].mean() if np.any(np.abs(coord-b)<.5) else 0 for b in bins])
                spec=np.abs(np.fft.rfft(prof-prof.mean()))
                if len(spec)>4:
                    k=int(np.argmax(spec[2:])+2); f=k/max(len(prof),1)
                    if .03<=f<=.35: freq[y:y+bs,x:x+bs]=f
        return ori,cv2.GaussianBlur(freq,(0,0),3)
    def _enhance(self,img,mask,ori,freq):
        src=img.astype(np.float32)/255; out=np.zeros_like(src); angles=np.linspace(0,math.pi,16,endpoint=False)
        valid=freq[freq>0]; median=float(np.median(valid)) if valid.size else .1; lam=float(np.clip(1/max(median,1e-3),6,30))
        for theta in angles:
            k=cv2.getGaborKernel((25,25),self.gabor_sigma,theta,lam,.5,0,cv2.CV_32F); resp=np.abs(cv2.filter2D(src,cv2.CV_32F,k))
            delta=np.abs(np.angle(np.exp(1j*(ori-theta)))); out=np.maximum(out,resp*np.exp(-(delta**2)/(2*(math.pi/18)**2)))
        out=cv2.normalize(out,None,0,255,cv2.NORM_MINMAX).astype(np.uint8); out[mask==0]=0
        # Fine local contrast + edge-preserving denoise while retaining ridge geometry.
        out=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(out)
        out=cv2.detailEnhance(cv2.cvtColor(out,cv2.COLOR_GRAY2BGR),sigma_s=10,sigma_r=.15)[:,:,0]
        return out
    def _skeleton(self,enh,mask):
        _,bw=cv2.threshold(enh,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU); bw[mask==0]=0
        sk=remove_small_objects(skeletonize(bw>0),min_size=12,connectivity=2); return sk.astype(np.uint8)*255
    def _cn(self,s,y,x):
        p=(s[y-1:y+2,x-1:x+2]>0).astype(np.uint8); ring=[p[0,0],p[0,1],p[0,2],p[1,2],p[2,2],p[2,1],p[2,0],p[1,0]]
        return int(sum(ring[i]!=ring[(i+1)%8] for i in range(8))/2)
    def _candidate(self,s,y,x,typ):
        p=(s[y-4:y+5,x-4:x+5]>0); d=float(p.mean())
        return .01<d<.42 if typ=="ending" else .02<d<.50
    def _extract(self,s,ori,mask):
        ys,xs=np.where(s>0); pts=[]; h,w=s.shape
        for y,x in zip(ys,xs):
            if x<self.border_margin or y<self.border_margin or x>=w-self.border_margin or y>=h-self.border_margin or mask[y,x]==0: continue
            cn=self._cn(s,y,x); typ="ending" if cn==1 else "bifurcation" if cn==3 else None
            if not typ or not self._candidate(s,y,x,typ): continue
            q=float(np.clip(1-abs(float(np.mean(s[y-4:y+5,x-4:x+5]>0))-.10),.1,1))
            pts.append({"x":int(x),"y":int(y),"type":typ,"angle":float(ori[y,x]),"quality":q})
        pts.sort(key=lambda a:a["quality"],reverse=True); out=[]
        for p in pts:
            if all((p["x"]-q["x"])**2+(p["y"]-q["y"])**2>=self.min_distance**2 for q in out): out.append(p)
        return out
    def process(self,image):
        n=self._normalize(image); q=self._quality(n); mask=self._segment(n); ori,freq=self._orientation_frequency(n); enh=self._enhance(n,mask,ori,freq); sk=self._skeleton(enh,mask); pts=self._extract(sk,ori,mask)
        q.update({"foreground_ratio":float(np.mean(mask>0)),"frequency_coverage":float(np.mean(freq>0)),"minutiae_density":float(len(pts)/max(int(np.sum(mask>0)),1)*10000)})
        q["status"]="usable" if q["foreground_ratio"]>.10 and q["coherence"]>.08 else "review"
        return {"enhanced":enh,"mask":mask,"skeleton":sk,"overlay":self._overlay(enh,sk,pts),"minutiae":pts,"counts":{"total":len(pts),"endings":sum(m["type"]=="ending" for m in pts),"bifurcations":sum(m["type"]=="bifurcation" for m in pts)},"quality":q,"template":{"format":"starfish-minutiae","version":1,"image":{"width":int(n.shape[1]),"height":int(n.shape[0])},"quality":q,"minutiae":pts}}
    def _overlay(self,base,sk,pts):
        c=cv2.cvtColor(base,cv2.COLOR_GRAY2BGR); c[sk>0]=(210,210,210)
        for m in pts:
            col=(0,0,255) if m["type"]=="ending" else (255,0,0); x,y,a=m["x"],m["y"],m["angle"]; cv2.circle(c,(x,y),4,col,1,cv2.LINE_AA); cv2.line(c,(x,y),(x+int(10*math.cos(a)),y+int(10*math.sin(a))),col,1,cv2.LINE_AA)
        return c
