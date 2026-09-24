import {useEffect,useRef,useState} from "react";
import {createRoot} from "react-dom/client";
import {mountScanScene,type ScanScene} from "./three";
import "./styles.css";
type Result={job_id:string;width:number;height:number;quality:{foreground_ratio:number;status:string;ridge_energy:number;coherence:number;frequency_coverage:number;minutiae_density:number};counts:{total:number;endings:number;bifurcations:number};minutiae:Array<{x:number;y:number;type:string;angle:number;quality:number}>;artifacts:Record<string,string>};
const API=(import.meta.env.VITE_API_URL||"").replace(/\/$/,"");
const steps=["Upload","Decode","Normalize","Segment","Ridge flow","Enhance","Skeletonize","Minutiae","Template","Artifacts"];
async function prepareImage(file:File):Promise<File>{
 const MAX=3.2*1024*1024;if(file.size<=MAX)return file;
 const img=new Image(),src=URL.createObjectURL(file);
 try{
  await new Promise<void>((resolve,reject)=>{img.onload=()=>resolve();img.onerror=()=>reject(new Error("Could not read image."));img.src=src});
  const scale=Math.min(1,1400/Math.max(img.naturalWidth,img.naturalHeight),Math.sqrt(MAX/file.size));
  const canvas=document.createElement("canvas");canvas.width=Math.max(1,Math.floor(img.naturalWidth*scale));canvas.height=Math.max(1,Math.floor(img.naturalHeight*scale));
  canvas.getContext("2d")!.drawImage(img,0,0,canvas.width,canvas.height);
  const blob=await new Promise<Blob|null>(resolve=>canvas.toBlob(resolve,"image/jpeg",.86));
  if(!blob||blob.size>MAX)throw new Error("Image remains too large after compression.");
  return new File([blob],"starfish-input.jpg",{type:"image/jpeg"});
 }finally{URL.revokeObjectURL(src)}
}
function App(){
 const [file,setFile]=useState<File|null>(null),[r,setR]=useState<Result|null>(null),[busy,setBusy]=useState(false),[progress,setProgress]=useState(0),[stage,setStage]=useState("Ready"),[err,setErr]=useState("");
 const canvas=useRef<HTMLCanvasElement|null>(null),scene=useRef<ScanScene|null>(null);
 useEffect(()=>{if(canvas.current)scene.current=mountScanScene(canvas.current);return()=>scene.current?.dispose()},[]);
 async function run(){
  if(!file||busy)return;setBusy(true);setR(null);setErr("");setProgress(4);setStage(steps[0]);let timer:ReturnType<typeof setInterval>|undefined;let tick=0;
  try{
   const prepared=await prepareImage(file);setStage(steps[1]);setProgress(12);
   timer=setInterval(()=>{tick=Math.min(tick+1,8);setProgress(p=>Math.min(p+9,86));setStage(steps[tick+1]||"Processing");scene.current?.frame(tick/9,"scanning")},420);
   const f=new FormData();f.append("file",prepared);
   const x=await fetch(API+"/api/v1/extract",{method:"POST",body:f});
   if(!x.ok){let message="Analysis request failed";try{const body=await x.json();message=body.detail||message}catch{}throw new Error(message)}
   const data=await x.json() as Result;setR(data);setProgress(100);setStage("Complete");scene.current?.frame(1,"done");
  }catch(e){setErr(e instanceof Error?e.message:"Analysis failed");setStage("Stopped safely");}
  finally{if(timer)clearInterval(timer);setBusy(false)}
 }
 const url=(p:string)=>p.startsWith("http")?p:API+p,dl=(k:string)=>API+"/api/v1/download/"+r!.job_id+"/"+k;
 return <main>
  <header className="top"><div className="brand">STARFISH</div><div className="top-meta">Fingerprint imaging &amp; minutiae analysis</div></header>
  <section className="hero"><div className="eyebrow">ANALYZE • VISUALIZE • EXPORT</div><h1>Fingerprint analysis, <span>clearly.</span></h1><p className="lead">A compact research-oriented pipeline for image enhancement, ridge visualization and minutiae extraction.</p>
   <div className="workspace">
    <div className="upload-card"><div className="card-kicker">01 / INPUT</div><h2>Upload fingerprint image</h2><p>PNG, JPG or WEBP. Large files are compressed before transfer.</p>
     <label className="drop"><input type="file" accept="image/png,image/jpeg,image/webp" onChange={e=>{setFile(e.target.files?.[0]||null);setR(null);setErr("")}}/><span className="drop-icon">＋</span><strong>{file?file.name:"Choose image"}</strong><em>{file?((file.size/1024/1024).toFixed(2)+" MB selected"):"Tap to browse"}</em></label>
     <button className="primary-btn" disabled={!file||busy} onClick={run}><span>{busy?"Scanning…":"Analyze fingerprint"}</span><b>→</b></button>{err&&<div className="error"><strong>Analysis failed</strong><span>{err}</span></div>}</div>
    <div className="scan-card"><div className="card-kicker">02 / SCAN VISUALIZER</div><canvas ref={canvas} className="scan-canvas"/><div className="scan-readout"><span>{busy?"PROCESSING":"READY"}</span><strong>{progress}%</strong></div><div className="progress"><i style={{width:progress+"%"}}/></div><div className="scan-stage"><span>•</span>{stage}</div></div>
   </div>
  </section>
  <section className="process"><div className="section-head"><div><div className="eyebrow">03 / PIPELINE</div><h2>Processing trace</h2></div><span className="trust">Live UI state + real server result</span></div><div className="steps">{steps.map((s,i)=><div className={(i<Math.max(1,Math.ceil(progress/100*steps.length))?"on ":"")+"step"} key={s}><span>{String(i+1).padStart(2,"0")}</span><b>{s}</b></div>)}</div></section>
  {r&&<section className="results"><div className="section-head"><div><div className="eyebrow">04 / RESULT</div><h2>Analysis complete</h2></div><span className="job">JOB {r.job_id.slice(0,10).toUpperCase()}</span></div>
   <div className="stats">{[[r.counts.total,"Minutiae"],[r.counts.endings,"Endings"],[r.counts.bifurcations,"Bifurcations"],[((r.quality.foreground_ratio*100).toFixed(1)+"%"),"Foreground"],[r.quality.status,"Quality"]].map(([a,b])=><div className="stat" key={String(b)}><strong>{a}</strong><span>{b}</span></div>)}</div>
   <div className="viewer"><div className="viewer-top"><div><b>Enhanced fingerprint</b><span>{r.width}×{r.height}</span></div><div className="links"><a href={dl("enhanced")} download>Download</a><a href={dl("overlay")} download>Overlay</a></div></div><img className="primary" src={url(r.artifacts.enhanced)} alt="Enhanced fingerprint"/></div>
   <div className="grid">{["enhanced","mask","skeleton","overlay"].map(k=><figure key={k}><img src={url(r.artifacts[k])} alt={k}/><figcaption><b>{k}</b><a href={dl(k)} download>Download</a></figcaption></figure>)}</div>
   <div className="template-card"><div><small>EXTRACTED TEMPLATE</small><h2>Minutiae template</h2><p>Coordinates, type, orientation and quality metadata from the processing pipeline.</p></div><a className="download" href={dl("template")} download>Download JSON</a></div>
  </section>}
  <footer><span>STARFISH</span><span>Research and imaging workflow • not an identity conclusion</span></footer>
 </main>;
}
createRoot(document.getElementById("root")!).render(<App/>);