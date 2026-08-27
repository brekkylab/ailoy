## Picks
1. urn:li:person:j8mq4dhr — Iris Pemberton — leads the exact three-team scope the posting describes (training orchestration, GPU capacity, feature store) and still writes the scheduler code herself
2. urn:li:person:c7rv4bkt — Ellis Norwood — owns Ironvale's Kubernetes/PyTorch training platform solo, wants a full-time role in Tokyo (hybrid)
3. urn:li:person:v6hd2ptq — Piper Jessup — owns GPU scheduling, PyTorch job packaging, and the model registry at Nightjar Dynamics
4. urn:li:person:q3fw7bkp — Lane Osgood — runs job scheduling/GPU allocation for a Kubernetes training+serving platform, wants full-time on-site Tokyo — closest location match in the pool
5. urn:li:person:c5nq8jtv — Lane Ashby — sole owner of Cindershift's GPU scheduling + training pipeline + model registry, wants full-time hybrid Tokyo
6. urn:li:person:b6xr4tqm — Kai Lockhart — built the GPU-pooling rework and feature/artifact store at Quantile Labs, but is looking for contract work, not full-time

## Method

**Must-haves gated:** Kubernetes, MLOps, PyTorch (as skills — checked separately from
headline text) + `--city Tokyo`. `search --city Tokyo --skill Kubernetes --skill PyTorch
--skill MLOps` returned 13 rows. The first line of that search showed Kubernetes has three
spellings (`Kubernetes`, `Kubernetes (K8s)`, `kubernetes-admin`) — I re-ran with each
variant to make sure nobody was lost to spelling; only Hina Yoshida (`Kubernetes (K8s)`)
and Piper Ingram (`kubernetes-admin`) use the other spellings, and both are already in the
combined 13. A cross-check with `--mentions` instead of `--skill` surfaced one more row,
`urn:li:person:m9jd7vhs`, also named Kai Lockhart — same company urn `#1694752`, same
title, same start month as `urn:li:person:b6xr4tqm`. Same person, second (stale, no-contact)
profile — folded in, not counted separately.

**Nice-to-haves checked within the 13**, not used to narrow: `--mentions GPU`,
`--mentions "feature store"`, `--mentions capacity` on the Tokyo+skills set. This is what
separated the generic "ML Platform Engineer" headlines from the people who actually
described owning GPU scheduling or a feature store in their position text.

**Read all 13 side by side** (`read` with every id at once) to check tenure overlap,
`wants`, contact reachability, and whether the ownership claim in the headline held up in
the position description.

## Comparisons that mattered

- Iris Pemberton and Kai Lockhart's Quantile Systems/Quantile Labs are different companies
  (`#3133440` vs `#1694752`) despite similar names — not the same employer, just checked
  because of the name collision risk called out in the tool help.
- `years` and `naive_years` were equal for all 13 (no hidden concurrent-employment
  inflation in this set).
- Lane Ashby's 6.8-year tenure sums a Backend Engineer role (explicitly "No ML work in
  this role") plus 2.8 years as ML Platform Engineer — the platform-specific tenure is
  the shorter number, and I ranked her below people with longer platform-specific runs.
- Kai Lockhart's two profiles (`urn:li:person:b6xr4tqm`, updated 2026-07-11, contactable;
  `urn:li:person:m9jd7vhs`, updated 2025-10-22, no contact) describe the same job at the
  same company id — used the newer, contactable one and did not double-count him.

## Risks on the picks

- **Iris Pemberton** (Director, 10,001+ headcount) is a level above a hands-on IC hire;
  her own text ("still spend time in the design docs and occasionally the code",
  "driving as technical owner rather than a sponsor") argues she stays hands-on, but the
  title/company-size jump is worth confirming in screening. No `wants` row was recorded
  for her, so her arrangement preference is unstated.
- **Ellis Norwood** wants Hybrid in Tokyo; the posting is on-site — a real but small gap.
- **Piper Jessup** has no stated `wants` — open to work, but the arrangement she wants is
  unrecorded.
- **Lane Ashby**'s ML-platform-specific tenure (2.8 years, since 2023-01) is thinner than
  the others'; the 6.8-year headline number includes non-ML backend work.
- **Kai Lockhart** wants Contract, on-site Tokyo — the posting is full-time. Kept him at
  #6 for the strength of his GPU-scheduling and feature-store ownership, but this is a
  real mismatch to confirm before investing more time.

<!-- rejected -->

- **urn:li:person:l5nz8crt (Orion Brennan)** — highest `years` (10.3) in the search and
  would rank #1 on a naive years-sort. Rejected: he is CTO of a 5,000+ person company; his
  own summary frames his platform involvement as setting direction and making standardization
  calls, not owning the scheduler or feature store himself. Seniority and scope mismatch for
  an IC-level platform role.
- **urn:li:person:s9wk4bpr (Blake Merrick)** — matches all three must-have skills and has
  8.2 years. Rejected on two facts: his summary says the hands-on "Kubernetes and MLOps work
  belongs to the engineers on my team" (he reviews, doesn't own), and his `wants` row is
  "ML Engineer · Remote" — the posting is on-site Tokyo.
- **urn:li:person:f5pq2mhb (Nova Ingram)** — headline and position description echo the
  posting's own wording almost verbatim (GPU scheduling on Kubernetes / training
  orchestration for PyTorch / the feature store the modeling team reads from). Strong
  paper fit, but `contacts` is empty — "there is no way to reach this person" — so she
  cannot be shortlisted regardless of fit.
- **urn:li:person:2vqtrqt9 (陽菜 山田)** and **urn:li:person:x5w1f5gp (莉子 山本)** — both
  carry the three must-have skills and multi-year platform-adjacent tenure, but both have
  no `contacts` row ("no way to reach this person"). Excluded for unreachability, not fit.
- **urn:li:person:jvwogrmn (Piper Ingram)** — carries `kubernetes-admin` and PyTorch, but
  her own summary calls her an "entry-level ML engineer," her PyTorch endorsement count is
  2, and her platform involvement is described as covering for a short-staffed platform
  team on contract, not owning a platform. Not open to work either.
- **urn:li:person:8f2v0usn (陽菜 吉田)** — has all three must-have skills, but only 2.0
  years of total tenure, one job, and no language in the description suggesting she owns
  (rather than uses) the platform. Too thin against "experience owning a platform other
  teams depended on."
- **urn:li:person:m9jd7vhs (Kai Lockhart)** — second, stale (updated 2025-10-22) profile
  of the same person as pick #6 (`urn:li:person:b6xr4tqm`): same company urn `#1694752`,
  same title, same 2021-05 start date. No contact row. Not counted as a separate candidate.
