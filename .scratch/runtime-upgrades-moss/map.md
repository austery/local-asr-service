# Runtime Upgrade and MOSS Adoption Decision Map

Label: wayfinder:map
Status: open
Assignee: unassigned

## Destination

An approved implementation handoff selecting reproducible runtime versions, deciding whether MOSS fits English long-form multi-speaker transcription, and bounding safe retirement. The map is complete when the remaining implementation has no unresolved admission, compatibility, or scope decision.

## Notes

- Lei accepted the working roadmap on 2026-09-07. The [roadmap and source preflight](../../docs/plans/2026-09-07-runtime-upgrades-and-moss-evaluation.md) contain the evaluation defaults and delivery sequence; measured outcomes remain open. No second general plan-approval round is needed.
- Apply [Wayfinder](/Users/leipeng/Documents/global-skills/wayfinder/SKILL.md) and the accepted [Lightweight Local Speech Gateway Boundary](../../docs/ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md). Consult additional skills only when they materially help the selected decision.
- Prioritize English long-form multi-speaker usability. Preserve Qwen3 single-speaker, Paraformer Chinese multi-speaker, SenseVoice, and Apple Speech roles unless a later explicit decision changes them.
- Small isolated experiments may supply evidence. Production implementation, promotion, pushes, merges, and destructive cleanup are not authorized by charting this map.
- Local tracker: child files under `issues/`; `Status: open` is unclaimed. Claim by setting `Assignee` to the driving developer and `Status: claimed` before work. Append the answer under `## Answer`, set `Status: resolved`, and add a named context link below. Keep answers to five bullets, linking detailed evidence.
- Blocking is the local tracker's `Blocked by` convention. Scan children in numeric order; the frontier contains open, unassigned children whose blockers are all resolved. Use the title when referring to any ticket.
- Resolve one substantive decision per session. A rejected runtime candidate can lead to MOSS No-Go/Defer and does not permanently block the independent FunASR decision.
- The documentation branch `codex/runtime-upgrades-moss-roadmap` includes remote main through `2ff1a34db7d6999714b6df64a313c56e484bc171`. Work only in `/Users/leipeng/Documents/Projects/local-asr-service-worktrees/runtime-roadmap` or another isolated worktree; leave the running original main checkout and its environment untouched. A measured baseline is still pending.
- These map/ticket files are the maintained, versioned local tracker. Original untracked copies in the running checkout are historical drafts. Preserve its existing untracked `transcript.json`.

## Decisions so far

<!-- Empty: charting resolved no decision. Add only gists with links to resolved tickets. -->

## Not yet specified

Any additional compatibility or caller-transition decision exposed by the runtime/adoption evidence. Create a new ticket only when its question becomes precise and an independent approval or tracking boundary is needed; keep detailed implementation slicing for the handoff.

## Out of scope

Production delivery within this map; custom model conversion or diarization recovery frameworks; new model families beyond MOSS; further Qwen3-Sortformer optimization; automatic routing; replacing all existing specialists; broad refactors; routine cache deletion.
