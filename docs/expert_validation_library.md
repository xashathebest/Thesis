# Expert Validation library

The sidebar contains a read-only, empty research library. It is **disabled by default** and ships no clips, transcripts, quotations, identities, consultation dates or original timestamps. A consultation clip documents what an expert said; it does not prove that the 50-30-20 weighting is mathematically optimal. This library does not call or change grading or inference.

## Access and consent

Before enabling the library, the research team must verify the actual consent and recording terms for each clip, transcript, researcher and intended display audience; record the authorization outside this repository. A flag is only an operator assertion and cannot verify consent or authenticate an expert. Do not copy original recordings, clips, real manifests or access tokens into this repository or frontend assets.

The optional implementation is intended only for a **direct local workstation connection**: it requires a strong research token in the `Authorization` header, a loopback client and local Host header, and rejects forwarded headers. It returns no private metadata without the token and uses `Cache-Control: no-store`; the browser keeps the token in memory and clears it when leaving the view. This shared local bearer token does **not** authenticate individual researchers or provide institutional role management, auditable identity, or protection from a privileged user of the same machine. Do not expose it via a network, reverse proxy or public URL. For institutional or remote access, first replace this gate with reviewed per-user authentication, research-role authorization, protected transport, access logging and appropriate storage controls.

After consent and the local access model have been reviewed, an authorized operator may configure the server process with `EXPERT_LIBRARY_ENABLED=1`, `EXPERT_LIBRARY_SCOPE_VERIFIED=1`, a high-entropy `EXPERT_LIBRARY_RESEARCH_TOKEN` of at least 32 characters, an absolute `EXPERT_LIBRARY_MANIFEST` path, and an absolute `EXPERT_LIBRARY_MEDIA_ROOT` directory. Both private paths must resolve outside the repository. Store and distribute the token through an appropriate secret manager, not a checked-in file or URL. The existing server must be restarted for the new route to load; environment changes take effect for its process only. Leaving either enabling assertion unset keeps all library endpoints unavailable.

## Private manifest

The [empty template](expert_library_manifest.empty.json) contains only `schema_version: expert-library-v1` and an empty `clips` array. After scope review, create a private manifest outside the repository. Each entry needs:

| Field | Contract |
|---|---|
| `id`, `rule_id`, `title`, `research_question` | Stable clip ID uses letters, digits, underscore or hyphen; other fields are nonempty text. |
| `category` | `body_weighting`, `missing_head`, `missing_tail`, `belly_defects`, or `grade_distinctions`. |
| `consultation_date`, `reviewer` | Actual ISO date and reviewer, recorded only after consultation. |
| `original_filename`, `original_start_seconds`, `original_end_seconds` | Original filename and source span in seconds. These are displayed as a source reference, not a guessed URL. Access to the original recording is not provided by this implementation. |
| `clipped_filename`, `clipped_length_seconds` | Basename of a privately stored `.mp4` or `.webm` and duration in seconds. Files larger than 100 MiB are unavailable. |
| `claim_kind`, `claim_text` | `exact_quote` or `paraphrase`, and the real approved quote or paraphrase. |
| `researcher_interpretation` | Optional clearly separate interpretation; do not present it as an expert quote. |
| `validation_status` | `proposed`, `expert-confirmed`, or `experimentally-validated`, supported by actual evidence. |

Invalid or duplicate entries fail closed; missing media is shown as unavailable. The media endpoint resolves basenames inside the private media root and checks for path traversal. The application does not issue public media URLs, and an authorized clip must be fetched with the token before the browser plays it. The source timestamp is a text reference; linking to or streaming the original requires separate permission and a reviewed access design.

The token gate, consent flag, timestamps and status fields cannot independently establish the truth of the consultation or whether material may be displayed. No real consultation material or allowed display scope was provided for this implementation, so the acceptance test involving a real panelist and clip remains outstanding.
