# Results

Aggregate results written by the scripts in `analysis/`. Rates and differences are new lead awards per 1,000 index
papers; intervals are 95% confidence intervals from 1,000 replicates of a two-way bootstrap. In `between.csv` and
`japan_between.csv`, the rows `change` give the after-window minus the before-window difference and the ratio of the two
risk ratios, with intervals from the same replicates. For Japan, awards are KAKEN research grants.

| Script | Files | In the article |
|---|---|---|
| `between.py` | `between.csv`, `between_neartop.csv` | Results, Figure 3, Table S4 |
| `japan.py` | `japan_between.csv`, `japan_pairs.csv`, `japan_detection.csv`, `japan_audit.csv` | Results, Figure 3b, Tables S13–S15 |
| `same_scientist.py` | `curves.csv`, `hinge*.csv`, `pairs*.csv` | Figures 1 and 3, Supporting Figure S1, Tables S6–S8, S10 |
| `first_paper.py` | `first_paper*.csv` | Tables S11–S12, S23 |
| `checks.py` | `timing.csv`, `cumulative.csv`, `pair_checks.csv`, `standardized_order.csv`, `recall_by_group.csv`, `detection.csv` | Tables S3, S5, S8, S9 |
| `linkage_audit.py` | `linkage_audit.csv` | Table S1 |
| `bounds.py` | `bounds_*.csv` | Figure 3, Tables S16–S17 |
| `event_study.py` | `event_*.csv` | Results, Figures 2 and 3, Tables S18–S23 |
| `split.py` | `split_type_domain.csv` | Results, Table S24 |

`linkage_acknowledged.csv` (Table S1) and `linkage_recall.csv` (Table S2) were computed from all acknowledgments in
OpenAlex for the audited papers, which are not redistributed here. The output corroboration and recall rows of
`japan_audit.csv` restate `data/jp_audit_outputs.csv` (see [data/README.md](../data/README.md)).

## `bounds.py`

All rows compare the top 5% of journals with all other journals unless stated, and use the bootstrap replicates of
`between.py`, `same_scientist.py` and `first_paper.py`, so uncorrected rows repeat the published intervals.

| File | Contents |
|---|---|
| `bounds_detection.csv` | Paper windows with an acknowledged own award (`windows`) and those in which the linkage records a new lead award (`detected`), by group and window; for first top-journal papers and their comparison papers also with the broad linkage |
| `bounds_corrected.csv` | Rates, differences, risk ratios, the change from the before to the after window, and the ratio of the two risk ratios, uncorrected (`correction` = none) and with each group-window rate divided by its detection share (`detection`); in each replicate the shares are drawn from their binomial distributions given the counts |
| `bounds_relative.csv` | Relative measures and their upper 95% limits (`upper_limit`, from the replicates of the ratio; `upper_limit_over_rate`, the upper limit of the absolute measure divided by the comparison rate) against a benchmark of a 10% relative increase: the ratio of the two risk ratios minus 1 (between scientists), the risk ratio minus 1 (same scientist), and, for the first top-journal paper, the change and the after-window difference divided by the matched comparison rate in the after window and the ratio of the two risk ratios minus 1 |
| `bounds_frontier.csv` | Detection-corrected change for the first top-journal paper when its after-window detection share is lowered by 0 to 5 percentage points |
| `bounds_broad.csv` | First top-journal paper with outcomes and funding-history matching variables from the broad linkage |
| `bounds_named.csv` | *Nature*, *Science*, *Cell*, *NEJM*, *The Lancet*, *JAMA* and *PNAS* against all other journals: between scientists (before and after windows and their change), same-scientist pairs at least 24 months apart, and the first paper in these journals among researchers without an earlier one (from 2009), matched as in `first_paper.py` |

## `event_study.py`

Rates are proportions of trials (multiply by 1,000 for rates per 1,000 first top-journal papers or comparison papers);
amounts are US dollars (Japan: yen). `sample` labels the population, `variant` the cells (A or B), `cells` the time
unit of the cells (month or quarter), and `comparator` the comparison papers (all other journals or near-top
journals). Intervals: 95% percentile intervals of 1,000 two-way bootstrap replicates.

| File | Contents |
|---|---|
| `event_curves.csv` | Cumulative incidence of a new lead award by month 0–84 after the trial month, treated and comparison trials and their difference (`outcome` award: exit competing; award_noexit: exit not competing) |
| `event_horizons.csv` | At 12, 24, …, 84 months: cumulative incidence (`measure` cif), P(active and funded), exit, and mean amounts started (`dollars`), for new awards, awards with exit not competing, awards with the keys of `award_links.parquet` (`award_pub`), NIH competing renewal among holders of a renewable project (`renewal`), and cumulative amounts reaching a threshold (`dollars_ge_*`); absolute and relative differences |
| `event_before.csv` | Share of trials with any new award in months −24 to −1, 0–23 and 12–35 (no exit or censoring; trials observed through the window) |
| `event_sizes.csv` | Treated trials in the cohort and matched, trials, scientists, events, exits, comparison trials censored by a top-journal paper, trials followed past 36, 60 and through 84 months, and trials with an award after their exit date |
| `event_change.csv` | For each after window (months 12–35, 36–59, 60–83), trials observed through its end: rates before and after, the two differences, their change, the risk ratios and their ratio, and the change over the comparison rate; `correction` none or detection |
| `event_detection.csv` | Trial windows with an acknowledged own award and those in which the linkage records a new award, by arm and window, for the detection correction |
| `event_validation.csv` | Trials of 2017–2020 with the before window and months 12–35, beside the matched comparison of `first_paper.csv` |

`event_eligibility.csv` gives the number of eligible and matched first top-journal papers in the event study (Table
S18); it comes from the cohort build on the computing cluster, which uses person-level records not redistributed here.
