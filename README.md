# prestige-precedes-prize

Code, data, and aggregate results for "The prestige precedes the prize: Top-journal papers and new federal grants."

## Contents

| Directory | Contents |
|---|---|
| `analysis/` | Analysis, figure, and table scripts |
| `data/` | Analysis tables built from OpenAlex |
| `results/` | Aggregate results |
| `figures/` | Figures 1–3 |
| `tables/` | Supporting Tables S3–S12 |

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
