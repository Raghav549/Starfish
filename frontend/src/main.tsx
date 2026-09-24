import {useState} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App(){
  const [file,setFile]=useState<File|null>(null);
  const [result,setResult]=useState<any>(null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");

  async function analyze(){
    if(!file) return;
    setBusy(true); setError(""); setResult(null);
    const fd=new FormData(); fd.append("file",file);
    try{
      const res=await fetch(API+"/api/v1/extract",{method:"POST",body:fd});
      if(!res.ok) throw new Error(await res.text());
      setResult(await res.json());
    }catch(e){setError(e instanceof Error?e.message:"Analysis failed")}
    finally{setBusy(false)}
  }

  return <main>
    <header><strong>STARFISH</strong><span>Fingerprint Analysis Engine</span></header>
    <section className="hero">
      <h1>Fingerprint → minutiae.</h1>
      <p>Upload a real fingerprint image and inspect enhancement, skeletonization and detected ridge endings / bifurcations.</p>
      <label className="drop"><input type="file" accept="image/*" onChange={e=>setFile(e.target.files?.[0]||null)}/>{file?file.name:"Choose fingerprint image"}</label>
      <button disabled={!file||busy} onClick={analyze}>{busy?"Processing…":"Analyze fingerprint"}</button>
      {error&&<pre className="error">{error}</pre>}
    </section>
    {result&&<section>
      <div className="stats">
        <div><b>{result.counts.total}</b><span>Total</span></div>
        <div><b>{result.counts.endings}</b><span>Endings</span></div>
        <div><b>{result.counts.bifurcations}</b><span>Bifurcations</span></div>
        <div><b>{(result.quality.foreground_ratio*100).toFixed(1)}%</b><span>Foreground</span></div>
      </div>
      <div className="grid">{["enhanced","skeleton","overlay"].map((k)=><figure key={k}><img src={API+result.artifacts[k]}/><figcaption>{k}</figcaption></figure>)}</div>
      <details><summary>Minutiae JSON</summary><pre>{JSON.stringify(result.minutiae,null,2)}</pre></details>
    </section>}
  </main>
}
createRoot(document.getElementById("root")!).render(<App/>);