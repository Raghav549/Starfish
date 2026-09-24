import {useState} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";
const API=import.meta.env.VITE_API_URL||"http://localhost:8000";
function App(){
 const[file,setFile]=useState<File|null>(null),[r,setR]=useState<any>(),[busy,setBusy]=useState(false),[err,setErr]=useState("");
 async function run(){if(!file)return;setBusy(true);setErr("");try{const f=new FormData();f.append("file",file);const x=await fetch(API+"/api/v1/extract",{method:"POST",body:f});if(!x.ok)throw Error(await x.text());setR(await x.json())}catch(e){setErr(e instanceof Error?e.message:"Analysis failed")}finally{setBusy(false)}}
 const dl=(k:string)=>API+"/api/v1/download/"+r.job_id+"/"+k;
 return <main>
  <header><b>STARFISH</b><span>Advanced Fingerprint Analysis</span></header>
  <section className="hero"><small>ENHANCE • EXTRACT • EXPORT</small><h1>Sharper fingerprint analysis.</h1><p>High-clarity enhancement, ridge structure processing, minutiae detection and downloadable outputs from your uploaded image.</p>
   <label className="drop"><input type="file" accept="image/*" onChange={e=>setFile(e.target.files?.[0]||null)}/>{file?file.name:"Upload fingerprint image"}</label>
   <button disabled={!file||busy} onClick={run}>{busy?"Processing…":"Analyze fingerprint"}</button>{err&&<pre className="error">{err}</pre>}
  </section>
  {r&&<section className="results">
   <div className="stats">{[[r.counts.total,"Minutiae"],[r.counts.endings,"Endings"],[r.counts.bifurcations,"Bifurcations"],[(r.quality.foreground_ratio*100).toFixed(1)+"%","Foreground"],[r.quality.status,"Quality"]].map(([a,b])=><div><strong>{a}</strong><span>{b}</span></div>)}</div>
   <div className="viewer"><div className="viewer-top"><b>Enhanced fingerprint</b><div><a href={dl("enhanced")} download>Download image</a><a href={dl("overlay")} download>Download overlay</a></div></div><img className="primary" src={API+r.artifacts.enhanced}/></div>
   <div className="grid">{["enhanced","mask","skeleton","overlay"].map(k=><figure><img src={API+r.artifacts[k]}/><figcaption><span>{k}</span><a href={dl(k)} download>Download</a></figcaption></figure>)}</div>
   <div className="template-card"><div><small>EXTRACTED TEMPLATE</small><h2>Fingerprint code</h2><p>Structured minutiae coordinates, type, orientation and quality metadata.</p></div><a className="download" href={dl("template")} download>Download JSON code</a></div>
   <details><summary>View minutiae data</summary><pre>{JSON.stringify(r.minutiae,null,2)}</pre></details>
  </section>}
 </main>
}
createRoot(document.getElementById("root")!).render(<App/>);
