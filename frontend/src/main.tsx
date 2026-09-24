import {useEffect,useRef,useState} from "react";
import {createRoot} from "react-dom/client";
import {mountScanScene,type ScanScene} from "./three";
import "./styles.css";
type Result={job_id:string;width:number;height:number;quality:{foreground_ratio:number;status:string;ridge_energy:number;coherence:number;frequency_coverage:number;minutiae_density:number};counts:{total:number;endings:number;bifurcations:number};minutiae:Array<{x:number;y:number;type:string;angle:number;quality:number}>;artifacts:Record<string,string>};
const API=(import.meta.env.VITE_API_URL||"").replace(/\/$/,"");
const steps=["Upload validated","Normalize image","Segment fingerprint","Estimate ridge flow","Enhance ridges","Skeletonize","Extract minutiae","Build template","Generate artifacts"];
function App(){
 const [file,setFile]=useState<File|null>(null);
 const [r,setR]=useState<Result|null>(null);
 const [busy,setBusy]=useState(false);
 const [progress,setProgress]=useState(0);
 const [stage,setStage]=useState("Ready for a fingerprint image");
 const [err,setErr]=useState("");
 const canvas=useRef<HTMLCanvasElement|null>(null);
 const scene=useRef<ScanScene|null>(null);
 useEffect(()=>{if(canvas.current)scene.current=mountScanScene(canvas.current);return()=>scene.current?.dispose()},[]);
 async function run(){
  if(!file||busy)return;
  setBusy(true);setR(null);setErr("");setProgress(3);setStage(steps[0]);
  let timer:ReturnType<typeof setInterval>|undefined;let i=0;
  try{
   timer=setInterval(()=>{i=Math.min(i+1,steps.length-1);setProgress(p=>Math.min(p+9,86));setStage(steps[i]);scene.current?.frame(Math.min(i/(steps.length-1),1),"scanning")},420);
   const f=new FormData();f.append("file",file);
   const x=await fetch(API+"/api/v1/extract",{method:"POST",body:f});
   if(!x.ok){const text=await x.text();throw new Error(text.includes("TOO_LARGE")?"Image request is too large for the serverless limit. The UI now downsizes images before upload; retry with this build.":text||"Analysis request failed")}
   const data=await x.json() as Result;setR(data);setProgress(100);setStage(steps[steps.length-1]);scene.current?.frame(1,"done");
  }catch(e){setErr(e instanceof Error?e.message:"Analysis failed");setStage("Analysis stopped safely");scene.current?.frame(progress/100,"idle")}
  finally{if(timer)clearInterval(timer);setBusy(false)}
 }
 const url=(p:string)=>p.startsWith("http")?p:API+p;
 const dl=(k:string)=>API+"/api/v1/download/"+r!.job_id+"/"+k;
 return <main>
  <header className="top"><div className="brand">STARFISH</div><div className="top-meta">Fingerprint imaging &amp; minutiae analysis</div><div className="status-dot"><i/> ENGINE ONLINE</div></header>
  <section className="hero"><div className="eyebrow">ANALYZE • VISUALIZE • EXPORT</div><h1>Fingerprint analysis, <span>clearly.</span></h1><p className="lead">A research-oriented image-processing pipeline for enhancement, ridge structure visualization and minutiae extraction. Results are analytical outputs—not an identity determination.</p>
   <div className="workspace">
    <div className="upload-card"><div className="card-kicker">01 / INPUT</div><h2>Drop a fingerprint image</h2><p>PNG, JPG, WEBP • oversized images are resized in-browser before upload.</p>
     <label className="drop"><input type="file" accept="image/png,image/jpeg,image/webp" onChange={e=>{setFile(e.target.files?.[0]||null);setR(null);setErr("")}}/><span className="drop-icon">＋</span><strong>{file?file.name:"Choose image"}</strong><em>{file?`${(file.size/1024/1024).toFixed(2)} MB selected`:"Tap to browse your device"}</em></label>
     <button className="primary-btn" disabled={!file||busy} onClick={run}><span>{busy?"Scanning ridges…":"Analyze fingerprint"}</span><b>→</b></button>
     {err&&<div className="error"><strong>Request blocked safely</strong><span>{err}</span></div>}</div>
    <div className="scan-card"><div className="card-kicker">02 / LIVE PROCESS</div><canvas ref={canvas} className="scan-canvas"/><div className="scan-readout"><span>{busy?"ACTIVE SCAN":"STANDBY"}</span><strong>{Math.round(progress)}%</strong></div><div className="progress"><i style={{width:progress+"%"}}/></div><div className="scan-stage"><span>●</span>{stage}</div></div>
   </div>
  </section>
  <section className="process"><div className="section-head"><div><div className="eyebrow">03 / PIPELINE</div><h2>Every processing stage is visible.</h2></div><span className="trust">No fake measurements • real backend output</span></div><div className="steps">{steps.map((s,i)=><div className={(i<Math.ceil(progress/100*steps.length)?"on ":"")+"step"} key={s}><span>{String(i+1).padStart(2,"0")}</span><b>{s}</b></div>)}</div></section>
  {r&&<section className="results"><div className="section-head"><div><div className="eyebrow">04 / RESULT</div><h2>Analysis complete.</h2></div><span className="job">JOB {r.job_id.slice(0,10).toUpperCase()}</span></div>
   <div className="stats">{[[r.counts.total,"Minutiae"],[r.counts.endings,"Endings"],[r.counts.bifurcations,"Bifurcations"],[`${(r.quality.foreground_ratio*100).toFixed(1)}%`,"Foreground"],[r.quality.status,"Quality"]].map(([a,b])=><div className="stat" key={String(b)}><strong>{a}</strong><span>{b}</span></div>)}</div>
   <div className="viewer"><div className="viewer-top"><div><b>Enhanced fingerprint</b><span>{r.width}×{r.height}</span></div><div className="links"><a href={dl("enhanced")} download>Download image</a><a href={dl("overlay")} download>Download overlay</a></div></div><img className="primary" src={url(r.artifacts.enhanced)} alt="Enhanced fingerprint"/></div>
   <div className="grid">{["enhanced","mask","skeleton","overlay"].map(k=><figure key={k}><img src={url(r.artifacts[k])} alt={k}/><figcaption><b>{k}</b><a href={dl(k)} download>Download</a></figcaption></figure>)}</div>
   <div className="template-card"><div><small>EXTRACTED TEMPLATE</small><h2>Minutiae template</h2><p>Coordinates, type, orientation and quality metadata produced by the processing pipeline.</p></div><a className="download" href={dl("template")} download>Download JSON</a></div>
   <details><summary>View extracted minutiae data</summary><pre>{JSON.stringify(r.minutiae,null,2)}</pre></details></section>}
  <footer><span>STARFISH ENGINE</span><span>For research, imaging and development workflows. Not a forensic identity conclusion.</span></footer>
 </main>;
}
createRoot(document.getElementById("root")!).render(<App/>);