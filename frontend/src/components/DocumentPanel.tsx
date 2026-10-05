import { FileText, Trash2, UploadCloud } from "lucide-react";
import { useRef, useState } from "react";
import type { DocumentRecord } from "../lib/api";

interface Props {
  documents: DocumentRecord[];
  userId: string;
  isAdmin: boolean;
  onUpload: (file: File, groups: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}

export function DocumentPanel({ documents, userId, isAdmin, onUpload, onDelete }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [groups, setGroups] = useState("");

  const accept = async (file?: File) => {
    if (!file) return;
    setBusy(true);
    try {
      await onUpload(file, groups);
    } finally {
      setBusy(false);
    }
  };

  return (
    <aside className="document-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Knowledge base</span>
          <h2>Sources</h2>
        </div>
        <span className="count">{documents.length}</span>
      </div>
      <button className="upload" disabled={busy} onClick={() => input.current?.click()}>
        <UploadCloud size={18} /> {busy ? "Uploading…" : "Add a document"}
      </button>
      <input
        ref={input}
        hidden
        type="file"
        accept=".pdf,.txt,.md"
        onChange={(event) => accept(event.target.files?.[0])}
      />
      <p className="upload-hint">PDF, TXT or Markdown · 25 MB max</p>
      <label className="share-label">Share with groups (optional)<input value={groups} onChange={(e) => setGroups(e.target.value)} placeholder="engineering, hr" /></label>
      <p className="upload-hint">Empty means owner and workspace administrators only.</p>
      <div className="document-list">
        {documents.length === 0 ? (
          <div className="empty-small">Your indexed documents will appear here.</div>
        ) : (
          documents.map((document) => (
            <div className="document-card" key={document.id}>
              <FileText size={18} />
              <div>
                <strong title={document.filename}>{document.filename}</strong>
                <span>v{document.version} · {document.chunk_count} chunks · {document.allowed_groups.join(", ") || "Private"}</span>
              </div>
              {(isAdmin || document.owner_id === userId) && <button aria-label={`Delete ${document.filename}`} onClick={() => onDelete(document.id)}>
                <Trash2 size={15} />
              </button>}
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
