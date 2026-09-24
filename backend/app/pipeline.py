from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.morphology import remove_small_objects, skeletonize

@dataclass
class FingerprintPipeline:
    block_size:int=16
    gabor_sigma:float=4.0
    min_distance:int=10
    border_margin:int=12
    min_component:int=24
    max_trace:int=28
    ridge_step_px:float=1.0
    target_ridge_wavelength:float=10.0
    reconstruction_sigma:float=3.0
    reconstruction_blend:float=0.82

    def _normalize(self,img):
        x=img.astype(np.float32); p1,p99=np.percentile(x,(1,99))
        return np.clip((x-p1)/max(p99-p1,1.0)*255,0,255).astype(np.uint8)

    def _quality(self,img):
        gx=cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)
        gxx=cv2.GaussianBlur(gx*gx,(0,0),3); gyy=cv2.GaussianBlur(gy*gy,(0,0),3); gxy=cv2.GaussianBlur(gx*gy,(0,0),3)
        coh=np.sqrt((gxx-gyy)**2+4*gxy**2)/(gxx+gyy+1e-6)
        energy=cv2.GaussianBlur(gx*gx+gy*gy,(0,0),5)
        return {"ridge_energy":float(np.mean(energy)),"coherence":float(np.mean(coh)),"coherence_p50":float(np.median(coh))}

    def _segment(self,img):
        x=img.astype(np.float32)
        mean=cv2.GaussianBlur(x,(0,0),7); mean2=cv2.GaussianBlur(x*x,(0,0),7)
        var=np.maximum(mean2-mean*mean,0)
        gx=cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)
        energy=cv2.GaussianBlur(gx*gx+gy*gy,(0,0),7)
        v=cv2.normalize(var,None,0,1,cv2.NORM_MINMAX); e=cv2.normalize(energy,None,0,1,cv2.NORM_MINMAX)
        score=.55*v+.45*e; cutoff=max(float(np.percentile(score,35)),.04)
        mask=(score>cutoff).astype(np.uint8)
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k); mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        comps=[(int(stats[i,cv2.CC_STAT_AREA]),i) for i in range(1,n) if int(stats[i,cv2.CC_STAT_AREA])>=max(self.min_component,int(mask.size*.001))]
        out=np.zeros_like(mask)
        for _,i in sorted(comps,reverse=True)[:max(1,min(3,len(comps)))]: out[labels==i]=1
        return out*255

    def _orientation_frequency(self,img,mask):
        h,w=img.shape; bs=self.block_size
        gx=cv2.Sobel(img.astype(np.float32),cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(img.astype(np.float32),cv2.CV_32F,0,1,ksize=3)
        gxx=cv2.GaussianBlur(gx*gx,(0,0),3); gyy=cv2.GaussianBlur(gy*gy,(0,0),3); gxy=cv2.GaussianBlur(gx*gy,(0,0),3)
        theta=.5*np.arctan2(2*gxy,gxx-gyy+1e-6)+math.pi/2
        coherence=np.sqrt((gxx-gyy)**2+4*gxy**2)/(gxx+gyy+1e-6)
        freq=np.zeros_like(theta,dtype=np.float32)
        for y in range(0,h-bs+1,bs):
            for x in range(0,w-bs+1,bs):
                if np.mean(mask[y:y+bs,x:x+bs]>0)<.55: continue
                a=float(np.median(theta[y:y+bs,x:x+bs])); block=img[y:y+bs,x:x+bs].astype(np.float32)
                yy,xx=np.indices(block.shape); coord=xx*math.cos(a)+yy*math.sin(a)
                bins=np.arange(-bs,bs+1,dtype=np.float32)
                profile=np.array([block[np.abs(coord-b)<.5].mean() if np.any(np.abs(coord-b)<.5) else 0 for b in bins],dtype=np.float32)
                profile-=profile.mean(); spec=np.abs(np.fft.rfft(profile))
                if len(spec)>4:
                    k=int(np.argmax(spec[2:])+2); f=k/max(len(profile),1)
                    if .03<=f<=.35: freq[y:y+bs,x:x+bs]=f
        return cv2.GaussianBlur(theta.astype(np.float32),(0,0),2),cv2.GaussianBlur(freq,(0,0),2),coherence

    def _ridge_reconstruction(self,img,mask,ori,freq):
        # Build a cleaner ridge/valley representation for visualization and downstream
        # skeletonization. Each valid pixel is filtered by its local orientation/frequency.
        src=img.astype(np.float32)/255.0
        valid=freq[(freq>0)&(mask>0)]
        default_freq=1.0/self.target_ridge_wavelength
        rf=float(np.median(valid)) if valid.size else default_freq
        wavelength=float(np.clip(1.0/max(rf,1e-3),6,18))
        bank=np.linspace(0,math.pi,32,endpoint=False)
        response=np.zeros_like(src)
        for a in bank:
            k=cv2.getGaborKernel((41,41),max(2.5,self.gabor_sigma),float(a),wavelength,.55,0,cv2.CV_32F)
            r=cv2.filter2D(src,cv2.CV_32F,k)
            delta=np.abs(np.angle(np.exp(1j*(ori-a))))
            w=np.exp(-(delta*delta)/(2*(math.pi/34)**2))
            response+=r*w
        response=cv2.normalize(response,None,0,1,cv2.NORM_MINMAX)
        # Combine response with normalized local image to avoid erasing legitimate ridge detail.
        base=cv2.normalize(src,None,0,1,cv2.NORM_MINMAX)
        out=(0.72*response+0.28*base)
        out[mask==0]=0
        return (np.clip(out,0,1)*255).astype(np.uint8)

    def _enhance(self,img,mask,ori,freq):
        valid=freq[(freq>0)&(mask>0)]; rf=float(np.median(valid)) if valid.size else .1
        wavelength=float(np.clip(1/max(rf,1e-3),6,24)); src=img.astype(np.float32)/255.; out=np.zeros_like(src)
        for a in np.linspace(0,math.pi,18,endpoint=False):
            kernel=cv2.getGaborKernel((31,31),self.gabor_sigma,float(a),wavelength,.5,0,cv2.CV_32F)
            response=cv2.filter2D(src,cv2.CV_32F,kernel)
            delta=np.abs(np.angle(np.exp(1j*(ori-a)))); weight=np.exp(-(delta**2)/(2*(math.pi/20)**2))
            out+=np.maximum(response,0)*weight
        out=cv2.normalize(out,None,0,255,cv2.NORM_MINMAX).astype(np.uint8); out[mask==0]=0
        out=cv2.createCLAHE(clipLimit=1.8,tileGridSize=(8,8)).apply(out)
        out=cv2.addWeighted(out,1.25,cv2.GaussianBlur(out,(0,0),.7),-.25,0); out[mask==0]=0
        return out

    def _binarize(self,enhanced,mask):
        win=max(15,self.block_size*2+1); th=threshold_sauvola(enhanced,window_size=win,k=.18,r=128)
        bw=((enhanced>th)&(mask>0)).astype(np.uint8)
        bw=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
        return cv2.morphologyEx(bw,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))

    def _neighbors(self,sk,y,x):
        out=[]; sk=sk>0
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy or dx:
                    yy,xx=y+dy,x+dx
                    if 0<=yy<sk.shape[0] and 0<=xx<sk.shape[1] and sk[yy,xx]: out.append((yy,xx))
        return out

    def _prune_spurs(self,sk,iters=2):
        sk=sk.copy()
        for _ in range(iters):
            remove=set()
            ys,xs=np.where(sk)
            for y,x in zip(ys,xs):
                n=self._neighbors(sk,int(y),int(x))
                if len(n)!=1: continue
                path=[(int(y),int(x))]; prev=None; cur=path[0]
                for _ in range(self.max_trace):
                    nbr=[p for p in self._neighbors(sk,*cur) if p!=prev]
                    if not nbr: break
                    nxt=nbr[0]; path.append(nxt)
                    if len(self._neighbors(sk,*nxt))!=2: break
                    prev,cur=cur,nxt
                if len(path)<7 and len(self._neighbors(sk,*path[-1]))>=3: remove.update(path[:-1])
            for y,x in remove: sk[y,x]=False
        return sk

    def _skeleton(self,enhanced,mask):
        sk=skeletonize(self._binarize(enhanced,mask)>0); sk=remove_small_objects(sk,min_size=16,connectivity=2)
        return self._prune_spurs(sk,2).astype(np.uint8)*255

    def _cn(self,sk,y,x):
        ring=[]
        for dy,dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
            yy,xx=y+dy,x+dx; ring.append(int(0<=yy<sk.shape[0] and 0<=xx<sk.shape[1] and sk[yy,xx]>0))
        return sum(abs(ring[i]-ring[(i+1)%8]) for i in range(8))/2

    def _trace(self,sk,y,x,a,steps):
        cur=(float(x),float(y)); angle=float(a); pts=[]
        for _ in range(steps):
            cx,cy=cur; ix,iy=int(round(cx)),int(round(cy))
            if ix<1 or iy<1 or ix>=sk.shape[1]-1 or iy>=sk.shape[0]-1: break
            nbr=self._neighbors(sk,iy,ix)
            if not nbr: break
            best=None
            for ny,nx in nbr:
                aa=math.atan2(ny-cy,nx-cx); d=abs(math.atan2(math.sin(aa-angle),math.cos(aa-angle)))
                cand=(d,nx,ny,aa); best=cand if best is None or cand[0]<best[0] else best
            _,nx,ny,angle=best; pts.append((nx,ny)); cur=(float(nx),float(ny))
        return pts

    def _local_quality(self,enhanced,coherence,y,x):
        patch=enhanced[max(0,y-5):y+6,max(0,x-5):x+6]
        return float(np.clip(.45*np.clip(np.std(patch)/64,0,1)+.55*np.clip(coherence[y,x],0,1),0,1))

    def _extract(self,sk,ori,mask,coherence,enhanced):
        ys,xs=np.where(sk>0); raw=[]; h,w=sk.shape
        for y,x in zip(ys,xs):
            y=int(y); x=int(x)
            if x<self.border_margin or y<self.border_margin or x>=w-self.border_margin or y>=h-self.border_margin or mask[y,x]==0: continue
            cn=self._cn(sk,y,x)
            if cn not in (1,3): continue
            a=float(ori[y,x]); fwd=len(self._trace(sk,y,x,a,self.max_trace)); back=len(self._trace(sk,y,x,a+math.pi,self.max_trace))
            if fwd<5 or back<3: continue
            raw.append({"x":x,"y":y,"type":"ending" if cn==1 else "bifurcation","angle":a,"quality":self._local_quality(enhanced,coherence,y,x),
                        "crossing_number":float(cn),"ridge_support_forward":fwd,"ridge_support_backward":back})
        raw.sort(key=lambda p:(p["quality"],p["ridge_support_forward"]+p["ridge_support_backward"]),reverse=True)
        out=[]
        for p in raw:
            if all((p["x"]-q["x"])**2+(p["y"]-q["y"])**2>=self.min_distance**2 for q in out): out.append(p)
        return out

    def _singular_points(self,ori,mask):
        h,w=ori.shape; bs=self.block_size; pts=[]
        for y in range(bs,h-bs,bs):
            for x in range(bs,w-bs,bs):
                if mask[y,x]==0: continue
                ring=[]
                for a in np.linspace(0,2*math.pi,16,endpoint=False):
                    yy=int(round(y+10*math.sin(a))); xx=int(round(x+10*math.cos(a)))
                    if 0<=yy<h and 0<=xx<w: ring.append(float(ori[yy,xx]))
                if len(ring)<8: continue
                idx=sum(.5*math.atan2(math.sin(2*(ring[(i+1)%len(ring)]-ring[i])),math.cos(2*(ring[(i+1)%len(ring)]-ring[i]))) for i in range(len(ring)))/math.pi
                if abs(idx)>=.5: pts.append({"x":x,"y":y,"type":"core_candidate" if idx>0 else "delta_candidate","poincare_index":float(idx)})
        # NMS on singular candidates to suppress repeated block hits.
        pts.sort(key=lambda p:abs(p["poincare_index"]),reverse=True); out=[]
        for p in pts:
            if all((p["x"]-q["x"])**2+(p["y"]-q["y"])**2>=16**2 for q in out): out.append(p)
        return out

    def _orientation_stats(self,ori,coherence,mask):
        valid=mask>0
        if not np.any(valid): return {"mean_orientation":0.0,"orientation_concentration":0.0,"orientation_std":0.0,"coherence_mean":0.0,"coherence_p10":0.0,"coherence_p50":0.0,"coherence_p90":0.0}
        angles=ori[valid]; coh=coherence[valid]; z=np.exp(1j*2*angles); mean=abs(np.mean(z))
        return {"mean_orientation":float(.5*math.atan2(np.mean(z).imag,np.mean(z).real)),"orientation_concentration":float(mean),
                "orientation_std":float(np.sqrt(max(0,-2*math.log(max(mean,1e-6))))/2),"coherence_mean":float(np.mean(coh)),
                "coherence_p10":float(np.percentile(coh,10)),"coherence_p50":float(np.percentile(coh,50)),"coherence_p90":float(np.percentile(coh,90))}


    def _ridge_quality_features(self, img, ori, freq, mask, enhanced):
        valid=mask>0
        if not np.any(valid):
            return {"ridge_valley_uniformity":0.0,"orientation_flow_mean":0.0,"orientation_certainty_mean":0.0,"frequency_mean":0.0}
        # NFIQ2-inspired diagnostics: frequency, local clarity, orientation certainty and flow.
        local_mean=cv2.GaussianBlur(enhanced.astype(np.float32),(0,0),3)
        local_var=cv2.GaussianBlur(enhanced.astype(np.float32)**2,(0,0),3)-local_mean**2
        clarity=float(np.mean(np.clip(np.sqrt(np.maximum(local_var,0))/64.0,0,1)[valid]))
        z=np.exp(1j*2*ori); certainty=float(np.mean(np.clip(np.abs(cv2.GaussianBlur(z.real,(0,0),2)+1j*cv2.GaussianBlur(z.imag,(0,0),2)),0,1)[valid]))
        flow=[]
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            shifted=np.roll(ori,(dy*self.block_size,dx*self.block_size),(0,1))
            d=.5*np.angle(np.exp(1j*2*(ori-shifted)))
            flow.append(np.abs(d))
        flow_mean=float(np.mean(np.stack(flow),axis=0)[valid].mean())
        fv=freq[valid & (freq>0)]
        freq_mean=float(np.mean(fv)) if fv.size else 0.0
        # Estimate ridge/valley width consistency through local gradients along the ridge-normal.
        gx=cv2.Sobel(enhanced,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(enhanced,cv2.CV_32F,0,1,ksize=3)
        normal_grad=np.abs(gx*np.cos(ori)+gy*np.sin(ori))
        rv_uniform=float(np.clip(1.0/(1.0+np.std(normal_grad[valid])/(np.mean(normal_grad[valid])+1e-6)),0,1))
        return {"local_clarity_mean":clarity,"orientation_certainty_mean":certainty,"orientation_flow_mean":flow_mean,
                "frequency_mean":freq_mean,"ridge_valley_uniformity":rv_uniform}

    def process(self,image):
        n=self._normalize(image); q=self._quality(n); mask=self._segment(n); ori,freq,coh=self._orientation_frequency(n,mask)
        reconstruction=self._ridge_reconstruction(n,mask,ori,freq); enh=self._enhance(reconstruction,mask,ori,freq); sk=self._skeleton(enh,mask); pts=self._extract(sk,ori,mask,coh,enh); singular=self._singular_points(ori,mask)
        q.update({"foreground_ratio":float(np.mean(mask>0)),"frequency_coverage":float(np.mean((freq>0)&(mask>0))),
            "ridge_frequency_median":float(np.median(freq[(freq>0)&(mask>0)])) if np.any((freq>0)&(mask>0)) else 0.0,
            "ridge_frequency_p10":float(np.percentile(freq[(freq>0)&(mask>0)],10)) if np.any((freq>0)&(mask>0)) else 0.0,
            "ridge_frequency_p90":float(np.percentile(freq[(freq>0)&(mask>0)],90)) if np.any((freq>0)&(mask>0)) else 0.0,
            "minutiae_density":float(len(pts)/max(int(np.sum(mask>0)),1)*10000),"singular_points":len(singular),**self._orientation_stats(ori,coh,mask),**self._ridge_quality_features(n,ori,freq,mask,enh)})
        q["status"]="usable" if q["foreground_ratio"]>.08 and q["coherence_p50"]>.18 and len(pts)>=4 else "review"
        return {"reconstruction":reconstruction,"enhanced":enh,"mask":mask,"skeleton":sk,"overlay":self._overlay(enh,sk,pts,singular),"minutiae":pts,"singular_points":singular,"orientation_field":ori,"ridge_frequency":freq,
                "counts":{"total":len(pts),"endings":sum(m["type"]=="ending" for m in pts),"bifurcations":sum(m["type"]=="bifurcation" for m in pts),"singular_points":len(singular)},
                "quality":q,"template":{"format":"starfish-minutiae","version":2,"image":{"width":int(n.shape[1]),"height":int(n.shape[0])},"quality":q,"minutiae":pts,"singular_points":singular,
                "ridge_frequency_summary":{"median":q["ridge_frequency_median"],"p10":q["ridge_frequency_p10"],"p90":q["ridge_frequency_p90"]}}}

    def _overlay(self,base,sk,pts,singular):
        c=cv2.cvtColor(base,cv2.COLOR_GRAY2BGR); c[sk>0]=(210,210,210)
        for m in pts:
            b=(0,0,255) if m["type"]=="ending" else (255,0,0); x,y,a=m["x"],m["y"],m["angle"]
            cv2.circle(c,(x,y),4,b,1,cv2.LINE_AA); cv2.line(c,(x,y),(x+int(10*math.cos(a)),y+int(10*math.sin(a))),b,1,cv2.LINE_AA)
        for p in singular:
            b=(0,215,255) if p["type"]=="core_candidate" else (255,0,255)
            cv2.drawMarker(c,(p["x"],p["y"]),b,cv2.MARKER_CROSS,14,1,cv2.LINE_AA)
        return c
