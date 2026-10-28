# prestige-precedes-prize

Code, data, and aggregate results for "Funding advantage precedes top-journal publication."

## Contents

| Directory | Contents |
|---|---|
| `analysis/` | Analysis, figure, and table scripts |
| `data/` | Analysis tables built from OpenAlex, including the NIH, NSF, and KAKEN (JSPS) grant records it distributes |
| `results/` | Aggregate results |
| `figures/` | Figures 1–3 and Supporting Figure S1 |
| `tables/` | Supporting Tables S3–S24 |

`analysis/japan.py` repeats the comparisons for Japanese-affiliated last authors and KAKEN research grants of the Japan
Society for the Promotion of Science (Figure 3b, Tables S13–S15). `analysis/bounds.py` gives the detection-corrected
and relative bounds (Figure 3, Tables S16–S17), `analysis/event_study.py` the first-paper event study (Figure 2, Tables
S18–S23), and `analysis/split.py` the comparison between scientists by kind of top journal and domain (Table S24).

Data details are in [data/README.md](data/README.md), and the results behind each figure and table in
[results/README.md](results/README.md).

## Installation

Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduction

```bash
./run_all.sh
```

On a 16-core node, `analysis/same_scientist.py` takes about 70 minutes and the other scripts a few minutes each.

## License

[MIT](LICENSE). OpenAlex data are released under CC0.

## Contact

Haining Wang

[hw56@iu.edu](mailto:hw56@iu.edu)
