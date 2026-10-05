import { Network, Quote } from "lucide-react";
import type { QueryResult } from "../lib/api";
import { api } from "../lib/api";
import { useEffect, useState } from "react";

export function AnswerView({ result }: { result: QueryResult }) {
  const [source, setSource] = useState("");
  const [error, setError] = useState("");
  useEffect(() => { setSource(""); setError(""); }, [result]);
  const openSource = async (id: string) => {
    try { setSource((await api.chunk(id)).text); setError(""); }
    catch (e) { setError(e instanceof Error ? e.message : "Could not load source"); }
  };
  return (
    <section className="answer-card">
      <div className="answer-meta">
        <span><Network size={15} /> Document-based response</span>
        <span className="confidence">{result.answer_mode} · retrieval score {result.retrieval_score.toFixed(2)}</span>
      </div>
      <p className="answer-text">{result.answer}</p>
      {result.citations.length > 0 && (
        <div className="citations">
          <h3>Evidence</h3>
          {result.citations.map((citation) => (
            <article key={citation.chunk_id}>
              <span className="citation-number">{citation.number}</span>
              <div>
                <button className="source-link" onClick={() => openSource(citation.chunk_id)}>{citation.filename} · v{citation.version}{citation.page ? ` · page ${citation.page}` : ""}</button>
                <p><Quote size={13} /> {citation.excerpt}</p>
              </div>
            </article>
          ))}
        </div>
      )}
      {source && <details open className="source-full"><summary>Full evidence passage</summary><p>{source}</p></details>}
      {error && <p className="error">{error}</p>}
    </section>
  );
}
