export interface DocumentRecord {
  id: string;
  filename: string;
  media_type: string;
  byte_size: number;
  created_at: string;
  status: string;
  chunk_count: number;
  owner_id: string;
  allowed_groups: string[];
  version: number;
}

export interface Citation {
  number: number;
  chunk_id: string;
  document_id: string;
  filename: string;
  excerpt: string;
  score: number;
  page?: number;
  section: string;
  version: number;
}

export interface QueryResult {
  answer: string;
  confidence: number;
  retrieval_score: number;
  answer_mode: string;
  citations: Citation[];
  trace?: Record<string, unknown>;
}

let token = "";
export const setAccessToken = (value: string) => { token = value; };
const request = (path: string, options: RequestInit = {}) => fetch(path, {
  ...options, headers: { ...options.headers, Authorization: `Bearer ${token}` },
});

const decode = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    const detail = payload.detail;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail || "Request failed"));
  }
  return response.json() as Promise<T>;
};

export const api = {
  session: () => request("/api/v1/session").then(decode<{user_id: string; workspace: string; groups: string[]; role: string}>),
  status: () => request("/api/v1/system").then(decode<{llm_model: string; llm_provider: string; graph_provider: string; counts: Record<string, number>}>),
  ready: () => request("/api/v1/ready").then(decode<{status: string}>),
  graph: () => request("/api/v1/graph").then(decode<GraphData>),
  trainingRuns: () => request("/api/v1/training-runs").then(decode<{runs: TrainingRun[]}>),
  chunk: (id: string) => request(`/api/v1/documents/chunks/${id}`).then(decode<{text: string; filename: string}>),
  listDocuments: () => request("/api/v1/documents").then(decode<DocumentRecord[]>),
  upload: (file: File, groups: string) => {
    const body = new FormData();
    body.append("file", file);
    body.append("allowed_groups", groups);
    return request("/api/v1/documents/jobs", { method: "POST", body }).then(
      decode<IngestionJob>,
    );
  },
  remove: (id: string) =>
    request(`/api/v1/documents/${id}`, { method: "DELETE" }).then((response) => {
      if (!response.ok) throw new Error("Could not delete the document");
    }),
  query: (question: string) =>
    request("/api/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 6, graph_hops: 1, include_debug: true }),
    }).then(decode<QueryResult>),
  job: (id: string) => request(`/api/v1/documents/jobs/${id}`).then(decode<IngestionJob>),
};

export interface IngestionJob { id: string; filename: string; status: string; stage: string; error?: string }
export interface GraphData { nodes: {id: string; name: string; kind: string}[]; edges: {source_id: string; target_id: string; relation_type: string; evidence_chunk_id: string; evidence_text: string}[] }
export interface TrainingRun { run_id: string; base_model: string; status: string; created_at: string; metrics: Record<string, number> }
