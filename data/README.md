# Analysis tables

Tables built from the 26 June 2026 OpenAlex snapshot, which includes the NIH ExPORTER and NSF Award Search records that
OpenAlex distributes as awards. Identifiers are OpenAlex IDs (`W…` works, `A…` authors, `S…` sources). Load the Parquet
files with `pandas.read_parquet`. The analysis code starts from these tables.

| File | Rows | Unit |
|---|---:|---|
| `index_papers.parquet` | 2,188,779 | Index paper: a US last author's earliest eligible last-authored paper of a year, 2014–2021 |
| `first_paper_cohort.parquet` | 1,118,144 | Index paper of 2017–2020, with the matching variables |
| `top_paper_dates.parquet` | 814,861 | Top-journal paper (first or last author) of a researcher in the cohort |
| `award_links.parquet` | 2,461,219 | Link between a last author and an award on which they are lead investigator |
| `link_audit.csv` | 700 | Audited link, with the language-model label and, for 120 links, the author's blind label |

Rows are stored in the order the analysis uses. The bootstrap assigns weights in the order in which scientists and
journals first appear, so reordering a file changes the intervals in the last digits.

## Provenance

- **Papers and topic-year sets.** English-language journal articles and reviews published 2014–2021. Titles were embedded
  with SPECTER2 and assigned to the nearest of 1,000 k-means centers. Papers whose Qwen3-Embedding-0.6B title embedding
  lay beyond a fixed distance from a reference topic taxonomy were excluded. A topic-year set is one cluster in one year.
- **Citation standing.** For a paper published in year *t*, the mean citations counted through *t*−1 to the journal's
  articles of *t*−3 to *t*−1. Within each set, journals with at least 100 such articles were ranked once each.
- **Index papers.** For each last author with a US affiliation in the publication year, the earliest eligible
  last-authored paper of each year.
- **Awards.** NIH research mechanisms (R01, R03, R15, R21, R24, R34, R35, R37, R56, DP1, DP2, U01), collapsed to the core
  project, and NSF Standard and Continuing Grants outside career-development, meeting, travel, fellowship, training,
  instrumentation, and center programs. An award is linked when its lead investigator matches a normalized name variant
  of the author and a US institution recorded for the author in the index year. Awards matching more than one author are
  dropped.

## `index_papers.parquet`

| Column | Meaning |
|---|---|
| `rid` | Row identifier |
| `work_id`, `author_id`, `journal_id` | The paper, its last author, and its journal |
| `choice_set_id` | Topic-year set (year × 1,000 + cluster) |
| `publication_date`, `publication_year` | Date of the index paper (time zero) |
| `first_last_date` | Date of the scientist's earliest last-authored paper |
| `pct`, `dec`, `quartile` | Percentile (0 to 1), decile, and quartile of the journal's citation standing in the set |
| `dom_name` | OpenAlex domain of the paper's primary topic (field) |
| `y_pre`, `y_post` | A new lead award starting in months −24 to <0 (before window) or +12 to <36 (after window) |
| `w2412`, `w1200`, `w0012`, `w0036` | The same in months −24 to <−12, −12 to <0, 0 to <12, and 0 to <36 |
| `f60` | A linked award started in the 60 months before the paper |
| `ref_pre`, `ref_post` | An award acknowledged in the author's own papers starts in the before or after window |
| `ref_awards`, `ref_linked` | Acknowledged own awards starting in months −24 to <36, and how many the linkage found |
| `first_top_date` | Date of the researcher's earliest top-journal paper (first or last author) from 2009 onward |

## `first_paper_cohort.parquet`

| Column | Meaning |
|---|---|
| `treated` | The journal is in the top 5% of its set |
| `L` | Years as last author at month −24 (`lt5`, `5to10`, `ge10`) |
| `funded_base` | A linked award started before month −24 |
| `career` | Years since the first OpenAlex paper (`0-4`, `5-9`, `10-19`, `20+`, `unknown`) |
| `log_works` | Log of 1 + OpenAlex works in calendar years *t*−5 to *t*−3 |
| `inst_band` | Institution type by tercile of research volume, from affiliations in years *t*−4 to *t*−2 |
| `recent_award` | An award started in months −60 to −24 |
| `near`, `near_early` | Highest percentile of citation standing among the researcher's first- or last-authored papers in the three calendar years before the index year, or in years *t*−5 to *t*−3 |
| `y_pre`, `y_post`, `y_w2412`, `y_w1200`, `y_w0012`, `y_w0036` | Outcomes, as in `index_papers.parquet` |

Other columns repeat those of `index_papers.parquet`.
