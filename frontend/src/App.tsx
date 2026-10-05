import { ArrowRight, BrainCircuit, Database, GitBranch, Search } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { AnswerView } from "./components/AnswerView";
import { DocumentPanel } from "./components/DocumentPanel";
import { GraphView } from "./components/GraphView";
import { api, setAccessToken, type DocumentRecord, type QueryResult, type IngestionJob, type GraphData, type TrainingRun } from "./lib/api";

export default function App() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [credential, setCredential] = useState("");
  const [signingIn, setSigningIn] = useState(false);
  const [session, setSession] = useState<{user_id: string; workspace: string; role: string} | null>(null);
  const [tab, setTab] = useState("ask");
  const [jobs, setJobs] = useState<IngestionJob[]>([]);
  const [graph, setGraph] = useState<GraphData>({nodes: [], edges: []});
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [modelStatus, setModelStatus] = useState("Checking local models…");
  const epoch = useRef(0);

  const refresh = async () => {
    const current = epoch.current;
    try {
      const [docs, connections] = await Promise.all([api.listDocuments(), api.graph()]);
      if (current === epoch.current) { setDocuments(docs); setGraph(connections); }
    } catch (e) { if (current === epoch.current) setError(e instanceof Error ? e.message : "Refresh failed"); }
  };
  useEffect(() => {
    if (!session) return;
    void refresh();
    const current = epoch.current;
    api.ready().then(() => { if (current === epoch.current) setModelStatus("Local models ready"); }).catch(() => { if (current === epoch.current) setModelStatus("Local models need setup"); });
    if (session.role === "admin") api.trainingRuns().then((value) => { if (current === epoch.current) setRuns(value.runs); }).catch(() => {});
  }, [session]);

  useEffect(() => {
    if (!session || !jobs.some((job) => ["queued", "running"].includes(job.status))) return;
    const current = epoch.current;
    const timer = setInterval(async () => {
      try {
        const updated = await Promise.all(jobs.map((job) => api.job(job.id)));
        if (current !== epoch.current) return;
        setJobs(updated);
        if (updated.some((job) => job.status === "completed")) void refresh();
      } catch (e) { if (current === epoch.current) setError(e instanceof Error ? e.message : "Could not poll ingestion"); }
    }, 2000);
    return () => clearInterval(timer);
  }, [jobs, session]);

  const login = async (event: FormEvent) => {
    event.preventDefault(); if (signingIn) return; setSigningIn(true); setError(""); setAccessToken(credential.trim());
    try { const user = await api.session(); epoch.current += 1; setSession(user); setCredential(""); }
    catch (e) { setAccessToken(""); setError(e instanceof Error ? e.message : "Sign in failed"); }
    finally { setSigningIn(false); }
  };

  const logout = () => {
    epoch.current += 1; setAccessToken(""); setSession(null); setDocuments([]);
    setJobs([]); setGraph({nodes: [], edges: []}); setRuns([]); setResult(null); setError(""); setBusy(false); setQuestion(""); setTab("ask");
  };

  const ask = async (event: FormEvent) => {
    event.preventDefault();
    if (question.trim().length < 3) return;
    setBusy(true);
    setError("");
    const current = epoch.current;
    try {
      const answer = await api.query(question.trim());
      if (current === epoch.current) setResult(answer);
    } catch (e) {
      if (current === epoch.current) setError(e instanceof Error ? e.message : "The query failed");
    } finally {
      if (current === epoch.current) setBusy(false);
    }
  };

  const upload = async (file: File, groups: string) => {
    setError("");
    const current = epoch.current;
    try {
      const job = await api.upload(file, groups);
      if (current === epoch.current) setJobs((previous) => [...previous, job]);
    } catch (e) {
      if (current === epoch.current) setError(e instanceof Error ? e.message : "Upload failed");
    }
  };

  const remove = async (id: string) => {
    if (!window.confirm("Delete this document and its indexed evidence?")) return;
    const current = epoch.current;
    try { await api.remove(id); if (current === epoch.current) { await refresh(); setResult(null); } }
    catch (e) { if (current === epoch.current) setError(e instanceof Error ? e.message : "Deletion failed"); }
  };

  if (!session) return <main className="login-shell"><form className="login-card" onSubmit={login}>
    <GitBranch size={35} /><span className="eyebrow">Company knowledge, on your infrastructure</span>
    <h1>ResearchGraph AI</h1><p>Sign in to your private workspace. Ask your administrator for an access token.</p>
    <label>Access token<input type="password" value={credential} onChange={(e) => setCredential(e.target.value)} autoComplete="off" disabled={signingIn} required /></label>
    <button className="upload" type="submit" disabled={signingIn}>{signingIn ? "Signing in…" : "Open workspace"} <ArrowRight size={16}/></button>
    {error && <p className="error">{error}</p>}<small>Credentials stay in this tab’s memory and clear when you sign out.</small>
  </form></main>;

  return (
    <main>
      <header>
        <a className="brand" href="#"><span><GitBranch size={20} /></span> ResearchGraph <b>AI</b></a>
        <div className="system-status">{modelStatus} · {session.workspace}<button onClick={logout}>Sign out</button></div>
      </header>
      <div className="shell">
        <DocumentPanel documents={documents} userId={session.user_id} isAdmin={session.role === "admin"} onUpload={upload} onDelete={remove} />
        <section className="workspace">
          <div className="hero">
            <span className="eyebrow">Your private knowledge workspace</span>
            <h1>Ask across your research,<br /><em>follow every connection.</em></h1>
            <p>Answers from the documents you can access, with evidence you can inspect.</p>
          </div>

          <nav className="tabs"><button className={tab === "ask" ? "active" : ""} onClick={() => setTab("ask")}>Ask</button><button className={tab === "graph" ? "active" : ""} onClick={() => setTab("graph")}>Connections</button>{session.role === "admin" && <button className={tab === "training" ? "active" : ""} onClick={() => setTab("training")}>Model customization</button>}</nav>
          {error && <div className="error">{error}</div>}
          {jobs.map((job) => <div className={`job ${job.status}`} key={job.id}>{job.filename} · {job.stage.replaceAll("_", " ")}{job.error && <span>{job.error}</span>}</div>)}
          {tab === "graph" && <GraphView graph={graph} />}
          {tab === "training" && <section className="training"><p>Training runs appear here after an administrator runs the local adapter workflow. New document uploads update retrieval without retraining.</p>{runs.length ? runs.map((run) => <article key={run.run_id}><strong>{run.run_id}</strong><p>{run.base_model} · {run.status}</p><pre>{JSON.stringify(run.metrics, null, 2)}</pre></article>) : <p className="muted">No training runs yet. See the training guide included with this installation.</p>}</section>}

          {tab === "ask" && <>
          <form className="query-box" onSubmit={ask}>
            <Search size={21} />
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="What do the documents say about…"
              rows={2}
            />
            <button disabled={busy || question.trim().length < 3}>
              {busy ? "Thinking…" : "Ask"} <ArrowRight size={17} />
            </button>
          </form>
          {result ? <AnswerView result={result} /> : (
            <div className="pipeline">
              <div><Database /><span><b>01</b><strong>Ingest</strong><small>Content-addressed chunks</small></span></div>
              <div><Search /><span><b>02</b><strong>Retrieve</strong><small>FAISS + BM25 fusion</small></span></div>
              <div><GitBranch /><span><b>03</b><strong>Expand</strong><small>Entity graph traversal</small></span></div>
              <div><BrainCircuit /><span><b>04</b><strong>Synthesize</strong><small>Grounded with citations</small></span></div>
            </div>
          )}
          </>}
        </section>
      </div>
    </main>
  );
}
