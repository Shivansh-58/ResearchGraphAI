import { useState } from "react";
import { api, type GraphData } from "../lib/api";

export function GraphView({ graph }: { graph: GraphData }) {
  const [evidence, setEvidence] = useState("");
  if (!graph.nodes.length) return <p className="muted">Upload a document to explore its connections.</p>;
  const names = new Map(graph.nodes.map((node) => [node.id, node.name]));
  return <section className="graph-view">
    <p className="muted">Only connections supported by documents you can access appear here.</p>
    {graph.edges.map((edge, index) => <button className="relation" key={`${edge.evidence_chunk_id}-${index}`} onClick={async () => {
      try { setEvidence((await api.chunk(edge.evidence_chunk_id)).text); }
      catch { setEvidence("Evidence is no longer accessible."); }
    }}><span>{names.get(edge.source_id)}</span><small>{edge.relation_type.replaceAll("_", " ")}</small><span>{names.get(edge.target_id)}</span></button>)}
    {evidence && <div className="source-full">{evidence}</div>}
  </section>;
}
