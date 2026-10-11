# Results

Aggregate results written by the scripts in `analysis/`. Rates and differences are new lead awards per 1,000 index
papers; intervals are 95% confidence intervals from 1,000 replicates of a two-way bootstrap.

| Script | Files | In the article |
|---|---|---|
| `between.py` | `between.csv`, `between_neartop.csv` | Results, Figure 3, Table S4 |
| `same_scientist.py` | `curves.csv`, `hinge*.csv`, `pairs*.csv` | Figures 1–3, Tables S6–S8, S10 |
| `first_paper.py` | `first_paper*.csv` | Figure 3, Tables S11–S12 |
| `checks.py` | `timing.csv`, `cumulative.csv`, `pair_checks.csv`, `standardized_order.csv`, `recall_by_group.csv`, `detection.csv` | Tables S3, S5, S8, S9 |
| `linkage_audit.py` | `linkage_audit.csv` | Table S1 |

`linkage_acknowledged.csv` (Table S1) and `linkage_recall.csv` (Table S2) were computed from all acknowledgments in
OpenAlex for the audited papers, which are not redistributed here.
