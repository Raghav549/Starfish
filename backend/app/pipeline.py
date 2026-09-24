from __future__ import annotations
import math
from dataclasses import dataclass
import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects

@dataclass
class FingerprintPipeline:
    block_size:int=16
    gabor_sigma:float=4.0
    min_distance:int=10
    border_margin:int=12
    min_component:int=24

    def _normalize(self,img):
        x=img.astype(np.float32)
        p1,p99=np.percentile(x,(1,99))
        x=np.clip((x-p1)/max(p99-p1,1)*255,0,255)
        return x.astype(np.uint8)

    def _quality(self,img):
        gx=cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3)
        gy=cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)
        energy=cv2.GaussianBlur(gx*gx+gy*gy,(0,0),5)
        coherence=[]
        bs=self.block_size
        for y in range(0,img.shape[0]-bs+1,bs):
            for x in range(0,img.shape[1]-bs+1,bs):
                a=gx[y:y+bs,x:x+bs]; b=gy[y:y+bs,x:x+bs]
                gxx=float(np.mean(a*a)); gyy=float(np.mean(b*b)); gxy=float(np.mean(a*b))
                coh=math.sqrt((gxx-gyy)**2+4*gxy*gxy)/(gxx+gyy+1e-6)
                coherence.append(coh)
        return {"mean_ridge_energy":float(np.mean(energy)),"mean_coherence":float(np.mean(coherence) if coherence else 0.0)}

    def _segment(self,img):
        gx=cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)
        energy=cv2.GaussianBlur(gx*gx+gy*gy,(0,0),7)
        thr=max(float(np.percentile(energy,45)),1e-4)
        mask=(energy>thr).astype(np.uint8)
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
        num,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        clean=np.zeros_like(mask)
        for i in range(1,num):
            if stats[i,cv2.CC_STAT_AREA]>=max(self.min_component,int(mask.size*0.002)):
                clean[labels==i]=1
        return clean*255

    def _orientation_frequency(self,img):
        bs=self.block_size
        h,w=img.shape
        ori=np.zeros((h,w),np.float32); freq=np.zeros((h,w),np.float32)
        gx=cv2.Sobel(img.astype(np.float32),cv2.CV_32F,1,0,ksize=3)
        gy=cv2.Sobel(img.astype(np.float32),cv2.CV_32F,0,1,ksize=3)
        gxx=cv2.GaussianBlur(gx*gx,(0,0),3); gyy=cv2.GaussianBlur(gy*gy,(0,0),3); gxy=cv2.GaussianBlur(gx*gy,(0,0),3)
        ori_block=0.5*np.arctan2(2*gxy,gxx-gyy+1e-6)+math.pi/2
        for y in range(0,h-bs+1,bs):
            for x in range(0,w-bs+1,bs):
                patch=img[y:y+bs,x:x+bs].astype(np.float32)
                theta=float(np.median(ori_block[y:y+bs,x:x+bs]))
                proj=(patch-patch.mean())*math.cos(theta)
                signal=np.mean(proj,axis=0)
                spec=np.abs(np.fft.rfft(signal))
                if len(spec)>3:
                    peaks=np.argsort(spec[2:])[-3:]+2
                    f=float(np.median(peaks)/max(len(signal),1))
                    if 0.03<=f<=0.35: freq[y:y+bs,x:x+bs]=f
                ori[y:y+bs,x:x+bs]=theta
        ori=cv2.GaussianBlur(ori,(0,0),3)
        freq=cv2.GaussianBlur(freq,(0,0),3)
        return ori,freq

    def _enhance(self,img,mask,orientation,frequency):
        src=img.astype(np.float32)/255.0
        angles=np.linspace(0,math.pi,16,endpoint=False)
        out=np.zeros_like(src)
        for theta in angles:
            lam=float(np.clip(1.0/max(float(np.median(frequency[frequency>0])) if np.any(frequency>0) else 0.1,1e-3),5,30))
            kernel=cv2.getGaborKernel((25,25),self.gabor_sigma,theta,lam,0.5,0,cv2.CV_32F)
            resp=np.abs(cv2.filter2D(src,cv2.CV_32F,kernel))
            delta=np.abs(np.angle(np.exp(1j*(orientation-theta))))
            weight=np.exp(-(delta**2)/(2*(math.pi/16)**2))
            out=np.maximum(out,resp*weight)
        out=cv2.normalize(out,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        out[mask==0]=0
        return out

    def _skeleton(self,enhanced,mask):
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(enhanced)
        _,bw=cv2.threshold(clahe,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bw[mask==0]=0
        sk=skeletonize(bw>0)
        sk=remove_small_objects(sk,min_size=12,connectivity=2)
        return (sk.astype(np.uint8)*255)

    def _cn(self,s,y,x):
        p=(s[y-1:y+2,x-1:x+2]>0).astype(np.uint8)
        ring=[p[0,0],p[0,1],p[0,2],p[1,2],p[2,2],p[2,1],p[2,0],p[1,0]]
        return int(sum(ring[i]!=ring[(i+1)%8] for i in range(8))/2)

    def _local_minutiae_quality(self,skel,y,x):
        patch=(skel[y-4:y+5,x-4:x+5]>0).astype(np.uint8)
        density=float(np.mean(patch))
        return float(np.clip(1.0-density*1.8,0.2,1.0))

    def _extract(self,skel,orientation,mask):
        ys,xs=np.where(skel>0); points=[]
        h,w=skel.shape
        for y,x in zip(ys,xs):
            if x<self.border_margin or y<self.border_margin or x>=w-self.border_margin or y>=h-self.border_margin or mask[y,x]==0: continue
            cn=self._cn(skel,y,x)
            typ="ending" if cn==1 else "bifurcation" if cn==3 else None
            if typ is None: continue
            q=self._local_minutiae_quality(skel,y,x)
            points.append({"x":int(x),"y":int(y),"type":typ,"angle":float(orientation[y,x]),"quality":q})
        return self._suppress(points)

    def _suppress(self,pts):
        accepted=[]
        for p in sorted(pts,key=lambda v:v["quality"],reverse=True):
            if all((p["x"]-q["x"])**2+(p["y"]-q["y"])**2>=self.min_distance**2 for q in accepted):
                accepted.append(p)
        return accepted

    def _template(self,minutiae,img_shape,quality):
        return {"version":1,"coordinate_system":"pixel","image":{"width":int(img_shape[1]),"height":int(img_shape[0])},"quality":quality,"minutiae":minutiae}

    def _overlay(self,base,skel,pts):
        c=cv2.cvtColor(base,cv2.COLOR_GRAY2BGR); c[skel>0]=(210,210,210)
        for m in pts:
            col=(0,0,255) if m["type"]=="ending" else (255,0,0)
            x,y=m["x"],m["y"]; a=m["angle"]
            cv2.circle(c,(x,y),4,col,1,cv2.LINE_AA)
            cv2.line(c,(x,y),(x+int(10*math.cos(a)),y+int(10*math.sin(a))),col,1,cv2.LINE_AA)
        return c

    def process(self,image):
        n=self._normalize(image)
        q=self._quality(n)
        mask=self._segment(n)
        orientation,frequency=self._orientation_frequency(n)
        enhanced=self._enhance(n,mask,orientation,frequency)
        skeleton=self._skeleton(enhanced,mask)
        minutiae=self._extract(skeleton,orientation,mask)
        q.update({"foreground_ratio":float(np.mean(mask>0)),"minutiae_density":float(len(minutiae)/max(np.sum(mask>0),1)*10000)})
        q["status"]="usable" if q["foreground_ratio"]>0.10 and q["mean_coherence"]>0.05 else "review"
        return {"enhanced":enhanced,"mask":mask,"skeleton":skeleton,"overlay":self._overlay(enhanced,skeleton,minutiae),
                "minutiae":minutiae,"counts":{"total":len(minutiae),"endings":sum(m["type"]=="ending" for m in minutiae),"bifurcations":sum(m["type"]=="bifurcation" for m in minutiae)},"quality":q,
                "template":self._template(minutiae,n.shape,q)}
