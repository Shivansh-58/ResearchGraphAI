# Privacy boundaries and operating limits

## Threat model

Protect company document contents from external model APIs and from users without read permission.
The host administrator, Docker daemon, company installation owner and model/credential provisioning
process are trusted. Malware/root access, stolen bearer tokens, malicious runtime images and
unreviewed model weights are outside the application access-control boundary.

## Implemented controls

- Authentication is required outside the explicit test environment. Tokens are random/high-entropy;
  the users file stores SHA256 digests and constant-time comparison is used.
- A principal is bound to one workspace. Administrators have access only within that workspace.
  Members can read their own documents or documents shared with their groups.
- Current-document permissions are applied before vector/BM25 index construction and every graph
  evidence lookup, not after a model has seen unauthorized material.
- Graph edges retain authorized source evidence. Company-scoped identifiers and deterministic
  node labels avoid hidden-document metadata being displayed as a public node label/type.
- Downloads, source passages, historical versions and jobs have permission checks. A group share
  does not grant deletion: deletion requires the owner or workspace administrator.
- Tokens stay in browser memory, not localStorage; sign-out clears displayed data/session state.
  Epoch checks prevent prior-session answer/upload responses being shown after a new sign-in.
- Runtime model URLs default to allowlisted local/internal HTTP hosts. No remote LLM provider,
  proxy-environment use, redirect-following, external fonts or browser analytics are configured.
- Docker exposes only a loopback web port, keeps API/model ports internal, uses an internal network,
  disables Ollama cloud features, and makes the API filesystem read-only except explicit storage.
- HF training/inference use local files, disable telemetry and remote model code. The trainer has
  `network_mode: none`. Company adapters require evaluated, approved, same-company deployment.
- Document contents, prompts and bearer tokens are not intentionally logged. API access logs are
  disabled; inference transport logs are suppressed. Model runtime/vendor logs must still be
  reviewed by the installation administrator.
- API responses are no-store with same-origin content policy, no referrer and nosniff headers.
  The Docker reverse proxy adds a basic request-rate limit.

## Important limitations

“No external API” is **not** the same as “fully secure.” Native mode does not enforce host network
isolation. A pre-existing model service must be independently checked for locality/cloud settings.
Verify an installation offline before uploading private documents; never keep the connected setup
network enabled for private operation. Hostnames explicitly added to the allowlist are trusted
operator choices; malicious configuration/host DNS can defeat a name-based policy.

Source-number checks and verbatim edge quotes do not prove semantic correctness. Small models can
hallucinate, mis-extract relations, miss aliases or follow document prompt injections. The system
separates evidence from instructions and abstains on invalid citations, but this is not a complete
prompt-injection defense. Use human review for consequential answers and add domain evals.

There is no disk encryption, application-managed key vault, enterprise SSO, immutable audit trail,
fine-grained adapter memorization control, automatic token expiry/rotation, malware scanner, OCR,
per-company process isolation or formal compliance certification. Use encrypted storage, restricted
OS permissions, approved backups and separate company deployments for actual private use.
The native API has no distributed rate limiter; upload storage/job count can grow and needs quota
management. A 25 MB request limit alone does not protect against all resource exhaustion.

Permission revocation takes effect on the next authenticated request; a request already underway
may complete. Copies already viewed/exported cannot be revoked. Deleting a document removes its
source/chunks/edges, but does not erase backup copies or knowledge memorized by an adapter.
Changing group-sharing policy requires a controlled update/new version; duplicates preserve the
existing policy rather than silently broadening permissions.
Model-name fingerprints detect embedding configuration changes, not a replacement of weights
under an unchanged mutable tag. Keep prepared embedding weights immutable; if replacing them,
use a fresh data directory and reingest. The offline bundle records and verifies blob hashes.

For LAN/multi-user production use, add TLS, enterprise authentication, token management, quotas,
secure audit events, dependency/image vulnerability scanning and a deployment-specific threat
review. Do not claim regulatory approval or “zero hallucinations” from the supplied tests.
