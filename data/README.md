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
| `award_links_broad.parquet` | 2,689,278 | The same, with the broad linkage |
| `named_paper_dates.parquet` | 116,760 | Publication date of a paper (first or last author) of a researcher in the cohort in one of seven named journals, 2009–2021 |
| `link_audit.csv` | 700 | Audited link, with the language-model label and, for 120 links, the author's blind label |
| `jp_index_papers.parquet` | 377,925 | Index paper: a Japanese-affiliated last author's earliest eligible last-authored paper of a year, 2014–2021 |
| `jp_award_links.parquet` | 541,613 | Link between a Japanese-affiliated last author and a KAKEN research grant of their researcher number |
| `jp_kaken_grants.parquet` | 523,435 | KAKEN research grant with a start date |
| `jp_link_audit.csv` | 600 | Audited Japanese link, with the language-model label |
| `jp_audit_outputs.csv` | 48 | Aggregate output corroboration and recall of the Japanese linkage |
| `event_trials.parquet` | 998,055 | Event study: a US last author's month with a ranked paper and no earlier top-journal paper, 2014–2021 (treated and matched comparison trials of three cohorts) |
| `event_awards.parquet` | 63,655 | New lead award starting in months −24 to 84 of an event-study trial |
| `event_ipcw_month.parquet`, `event_ipcw_quarter.parquet`, `event_ipcw_neartop.parquet` | 607,875; 840,671; 115,983 | Censoring weights of the comparison trials of each cohort |
| `jp_event_trials.parquet` | 104,186 | The same for Japanese-affiliated last authors |
| `jp_event_awards.parquet` | 74,701 | KAKEN research grant starting in months −24 to 84 of a Japanese trial |
| `jp_event_ipcw_month.parquet`, `jp_event_ipcw_quarter.parquet`, `jp_event_ipcw_neartop.parquet` | 50,075; 85,600; 7,421 | Censoring weights of the Japanese comparison trials |

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
  dropped. `award_links.parquet` (columns `author_id`, `publication_year`, `award_key`, `award_start`) holds the links
  from which the outcomes and funding variables are computed. The broad linkage (`award_links_broad.parquet`, same
  columns) accepts any US institution in the author's OpenAlex affiliations from five years before to three years after
  the index year, and dates an award from the earliest start recorded under the matched name and institution.
- **Named journals.** *Nature*, *Science*, *Cell*, *New England Journal of Medicine*, *The Lancet*, *JAMA*, and
  *Proceedings of the National Academy of Sciences*, identified by their OpenAlex source IDs (listed in
  `analysis/bounds.py`). `named_paper_dates.parquet` (`author_id`, `publication_date`) lists the articles and reviews in
  these journals, published 2009–2021, on which a researcher in `first_paper_cohort.parquet` is first or last author.

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

## Japan

The Japanese tables come from the same OpenAlex snapshot, which includes the Grants-in-Aid for Scientific Research
(KAKEN) of the Japan Society for the Promotion of Science (JSPS) and the Ministry of Education, Culture, Sports, Science
and Technology as awards.

- **Index papers.** As for the United States, for last authors with a Japanese institution among their OpenAlex
  affiliations in the publication year, with the same topic-year sets and journal percentiles.
- **Grants.** KAKEN research grants in the categories Scientific Research (S), (A), (B), and (C), Young Scientists
  (including Start-up), Early-Career Scientists, Challenging Exploratory Research, Challenging Research (Exploratory and
  Pioneering), Research Activity Start-up, and Specially Promoted Research; fellowship categories are excluded. A grant
  is identified by its grant number (`KAKEN:<number>`) and starts at its project start date.
- **Linkage through the researcher number.** Each grant records its lead investigator's KAKEN researcher number. An
  index paper is linked to a researcher number when a normalized name variant of the author and a Japanese institution
  recorded for the author in the index year (name or OpenAlex ID) match the lead investigator's name and an institution
  on any of that researcher's grants whose lead investigator name is recorded in Latin script. All research grants of
  the researcher number, including those recorded only in Japanese script, then link to the index paper. Researcher
  numbers matching more than one author are dropped. An author may link to more than one researcher number.
- **Audit.** For `jp_link_audit.csv`, Claude Haiku 4.5 labeled 150 outcome-window links (grant starting in months −24
  to <0 or +12 to <36) for each window and for top-journal and other index papers of 2014–2021, sampled by a
  deterministic hash. `jp_audit_outputs.csv` was computed from the output papers that KAKEN grants list and the
  OpenAlex authors of those papers, which are not redistributed here: *corroboration* is the share of outcome-window
  links whose researcher number lists outputs in which the linked author is an author of one of them; *recall* is the
  share of grants starting in the window, held by lead investigators identifiable from their outputs (exactly one
  output author carries the lead investigator's name), that the linkage attaches to the investigator's index paper.

### `jp_index_papers.parquet`

`rid`, `work_id`, `author_id`, `journal_id`, `choice_set_id`, `publication_date`, `publication_year`, `pct`,
`dom_name`, `y_pre`, `y_post`, and `f60` are defined as in `index_papers.parquet`, with KAKEN research grants as the
awards. `n_nrid` is the number of researcher numbers linked to the index paper.

### `jp_award_links.parquet`

`author_id`, `publication_year`, `award_key` (`KAKEN:<grant number>`), `award_start`.

### `jp_kaken_grants.parquet`

| Column | Meaning |
|---|---|
| `award_key`, `scheme`, `award_start` | Grant number, KAKEN category, project start date |
| `latin_name` | The lead investigator's name is recorded in Latin script |
| `reachable` | `latin_name`, or the lead investigator's researcher number holds a grant with a Latin-script name |
| `early_career` | Young Scientists, Early-Career Scientists, or Research Activity Start-up |

### `jp_link_audit.csv`

`window` (before or after), `paper` (top or other), the grant's `funder` category, title and start, the script of the
grant's lead investigator name, whether the author's name is distinctive, whether the index paper is linked to one or
more researcher numbers, whether the researcher number lists output papers and the author is on them, and the model's
label and confidence.

## Event study (`event_*.parquet`, `jp_event_*.parquet`)

Trials of last authors (scientist-months), built from all of a researcher's works rather than one index paper a year.
A trial is a month in which a last author with an affiliation in the country that year publishes a ranked paper (a
journal with a percentile in its topic-year set), has no earlier first- or last-authored top-journal paper from 2009
onward (rule as for `first_top_date`), and has an article or review in the previous 36 months. Treated trials publish
their first top-journal paper; comparison trials publish only other ranked papers that month. Comparison trials are
matched in exact cells and capped at 20 per treated trial in each cell (seeded). Three cohorts: cells with the calendar
month (`in_month`), with the calendar quarter (`in_quarter`), and with comparison papers restricted to near-top journals,
percentile 0.85 to <0.95 (`in_neartop`). Rows are in the order (scientist, month) used by the analysis.

| Column | Meaning |
|---|---|
| `author_id`, `journal_id`, `work_id`, `choice_set_id` | Scientist, journal and paper of the trial (earliest ranked paper of the month; for treated trials, the first top-journal paper) |
| `mo`, `publication_year` | Trial month (year × 12 + month − 1) and year |
| `treated`, `pct` | First top-journal paper; percentile of the trial paper's journal in its set |
| `in_month`, `in_quarter`, `in_neartop` | Member of each cohort |
| `cellA_*`, `cellB_*` | Cell codes in each cohort (−1 if not a member). A: month (or quarter), field, years as last author at month −24, an award before month −24, career age, productivity tercile, institution band, an award in months −60 to −24, highest percentile in years *t*−3 to *t*−1 (as in `first_paper_cohort.parquet`). B: A and any new award in months −24 to −1 |
| `dom_name`, `L`, `career`, `prod3`, `inst_band`, `near`, `funded_base`, `recent_award`, `award_pre24` | The prior record behind the cells |
| `pre_award_m24` | A new lead award started in months −24 to −1 |
| `T_award` | Months from the trial month to the first new lead award (NIH core project at its original start, or NSF award); empty if none by the snapshot |
| `T_award_pub` | The same with the award keys of `award_links.parquet` |
| `T_exit` | Months to exit: the scientist's last article or review that is followed by more than 24 months without one, dated at that paper |
| `T_top` | Comparison trials: months to the scientist's first top-journal paper (censoring) |
| `T_admin` | Months to the end of complete award coverage: June 2024 (US; the snapshot less 24 months) and April 2022 (Japan; KAKEN grants of fiscal 2022 onward are reached less completely); `T_admin_alt`: June 2024 for Japan |
| `ren_elig`, `T_ren` | US: the scientist holds a linked renewable NIH project (R01, R15, R24, R35, R37, U01) whose competitive segment covers the trial month; months to the first competing renewal (type 2) of a linked renewable project, dated the day after the previous segment's latest end date |
| `ref_pop`, `ref_m24_m1`, `ref_12_35`, `ref_36_59`, `ref_60_83` | US: the scientist is in the population of the acknowledgment reference set (index records in the top or bottom journal quartile); an award acknowledged in the scientist's own papers and led by them starts in months −24 to −1, 12–35, 36–59 or 60–83 |

`event_awards.parquet`: `author_id`, `mo` (the trial), `k` (months after the trial month), `amount` (NIH: OpenAlex
`amount` of the type-1 records of the project's first support year, supplements excluded, the first budget year; NSF:
the award amount; Japan: the KAKEN grant amount in yen; empty if not recorded).

`event_ipcw_*.parquet`: `author_id`, `mo`, and `q00`–`q28`, the stabilized inverse probability that a comparison trial
has not yet been censored by its own first top-journal paper, for months 3*q* to 3*q*+2 after the trial month (empty
after censoring). The weights were fitted on the cluster with `censoring_weights` in `analysis/event_study.py` (pooled
logistic regression over person-quarters, fitted on the first- and last-author cohort; the author-month panels it uses
are not redistributed), and are written here for the last-author comparison trials. Files stay under 95 MB; a file
split in parts would be named `*_1.parquet`, `*_2.parquet` in row order.
