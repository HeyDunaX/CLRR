# Difference-preserving prototype v1

Experimental CE + bilingual difference-direction cosine; optional original
CLRR. No old LSR, learned module or inference change. This is not an exact RKD
reproduction or a novelty/quality claim.

Full formula, audit, four-arm definitions, recipe paths and admission policy:
[`docs/research/DIFFERENCE_PRESERVING_PILOT.md`](../../../../docs/research/DIFFERENCE_PRESERVING_PILOT.md).

CPU tests use synthetic peers. Actual data admission requires reviewed train
pairs and a novelty decision; neither has passed. RTX training is blocked.

Mining policy v2 retains Unicode combining marks inside lexical tokens after
NFKC/casefold. It is a discovery heuristic, not linguistic word segmentation.
Candidate peers retain the deterministic minimum reviewed row-index policy;
not every graph edge is used. Generic controls exclude self, all reviewed
neighbors, and identical source/target strings. Audit JSON explicitly reports
conditional selected-edge coverage; no pair is admitted by an audit count.

Fresh corrected audits: `outputs/experiments/difference_preserving_feasibility/unicode_fix_v2/`.
Earlier frozen bundles/technical receipts belong to their original source;
source verification/rebuild is required before future deployment. Do not copy
old gate receipts into a new run.
