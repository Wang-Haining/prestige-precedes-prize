"""Between scientists: directly standardized differences for each definition of a top journal and against near-top
journals, before and after publication, and the smallest difference of interest (one quarter of the after-window
difference). -> results/between.csv, results/between_neartop.csv"""

import numpy as np
import pandas as pd

from analysis.common import YS, B, boot_w, cuts, load, save, std_gap

BANDS = {"near-top 0.85-0.95": 0.85, "near-top 0.90-0.95": 0.90}


def between(d):
    rows = []
    C = cuts(d)
    s = d.s.to_numpy()
    est = {(c, y): std_gap(s, A, d[y].to_numpy(float), np.ones(len(d))) for c, A in C.items() for y in YS}
    rng = np.random.default_rng(20261010)
    reps = {k: [] for k in est}
    for _ in range(B):
        w = boot_w(d, rng)
        for c, A in C.items():
            for y in YS:
                reps[(c, y)].append(std_gap(s, A, d[y].to_numpy(float), w)[:2])
    for (c, y), e in est.items():
        r = np.array(reps[(c, y)])
        rows.append(
            {
                "part": "between",
                "label": c,
                "window": y,
                "rd": e[0],
                "rd_lo": np.quantile(r[:, 0], 0.025),
                "rd_hi": np.quantile(r[:, 0], 0.975),
                "rr": e[1],
                "rr_lo": np.quantile(r[:, 1], 0.025),
                "rr_hi": np.quantile(r[:, 1], 0.975),
                "rate_top": e[2],
                "rate_non": e[3],
                "papers_top": int(C[c].sum()),
                "papers": len(d),
            }
        )
    for c in C:
        rows.append({"part": "delta", "label": c, "delta_per_1000": est[(c, "y_post")][0] / 4})
    save(rows, "between.csv")


def between_neartop(d):
    rng = np.random.default_rng(20261014)
    rows = []
    for lab, lo in BANDS.items():
        s = d[d.pct >= lo].reset_index(drop=True)
        A = (s.pct >= 0.95).to_numpy()
        si = pd.factorize(s.s)[0]
        est = {y: std_gap(si, A, s[y].to_numpy(float), np.ones(len(s))) for y in YS}
        reps = {y: [] for y in YS}
        aa, jj = pd.factorize(s.a)[0], pd.factorize(s.j)[0]
        for _ in range(B):
            w = rng.exponential(1.0, aa.max() + 1)[aa] * rng.exponential(1.0, jj.max() + 1)[jj]
            for y in YS:
                reps[y].append(std_gap(si, A, s[y].to_numpy(float), w)[:2])
        for y in YS:
            r = np.array(reps[y])
            e = est[y]
            rows.append(
                {
                    "part": "between",
                    "label": lab,
                    "window": y,
                    "rd": e[0],
                    "rd_lo": np.quantile(r[:, 0], 0.025),
                    "rd_hi": np.quantile(r[:, 0], 0.975),
                    "rr": e[1],
                    "rr_lo": np.quantile(r[:, 1], 0.025),
                    "rr_hi": np.quantile(r[:, 1], 0.975),
                    "rate_top": e[2],
                    "rate_non": e[3],
                    "papers_top": int(A.sum()),
                    "papers": len(s),
                }
            )
        rows.append({"part": "delta", "label": lab, "delta_per_1000": est["y_post"][0] / 4})
    save(rows, "between_neartop.csv")


if __name__ == "__main__":
    d = load()
    between(d)
    between_neartop(d)
