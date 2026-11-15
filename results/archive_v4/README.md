# Results under the v4 protocol (superseded)

These results were produced before the external review of November 2026 and are
kept for transparency. They are **not** used in the paper. Differences to v5:

* ISO-NE load models were fit once (training ended 2024-07-04) and scored on all of
  2025 without refits; v5 refits every model before each of 13 rolling windows.
* Tree ensembles did not see the load at the origin or the weather path; Chronos
  and the weekly naive forecast were labelled set B instead of B+.
* ISO-NE price models used observed load of the whole day D-1, which is not known
  at the 10:00 bidding gate (leakage); v5 restricts it to the gate.
* `deep/` holds the single-split deep-model runs and the five-seed study of v4.

The EPF market results (`results/price_epf_*`) are unaffected by these changes
and were kept at the top level.
