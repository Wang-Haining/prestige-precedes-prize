"""First top-journal paper, index years 2017-2020: researchers with no earlier top-journal paper (first or last author,
from 2009) whose index paper is their first in a top journal, against matched researchers whose index paper of the
same year appeared elsewhere (or, with NEAR=1, in a near-top journal). Exact cells on index year, field, years as last
author and any award before month -24; coarsened exact matching on career age, prior productivity, institution type
and volume, an award in months -60 to -24, and the highest earlier percentile of citation standing.
-> results/first_paper.csv, first_paper_neartop.csv, first_paper_checks.csv"""

import numpy as np
import pandas as pd

from analysis.common import DATA, B, att, cells, save

STRATA = ["publication_year", "dom_name", "L", "funded_base"]
MATCH = ["career", "prod3", "inst_band", "recent_award", "near"]


def load():
    d = pd.read_parquet(DATA / "first_paper_cohort.parquet")
    d["prod3"] = d.groupby("publication_year").log_works.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False)).astype(int)
    pub = pd.to_datetime(d.publication_date)
    ft = pd.to_datetime(d.first_top_date)
    ordinary = ft.isna() | (ft >= pub)
    return d, ordinary


def run(near_top=False):
    d, ordinary = load()
    # comparison researchers who published a top-journal paper within the next 36 months
    td = pd.read_parquet(DATA / "top_paper_dates.parquet")
    td["publication_date"] = pd.to_datetime(td.publication_date)
    x = d.loc[ordinary & d.treated.eq(0), ["rid", "author_id", "publication_date"]].merge(td, on="author_id", suffixes=("", "_top"))
    x = x[
        (x.publication_date_top > pd.to_datetime(x.publication_date))
        & (x.publication_date_top < pd.to_datetime(x.publication_date) + pd.DateOffset(months=36))
    ]
    later = d.rid.isin(x.rid).to_numpy()
    if near_top:
        keep = d.treated.eq(1).to_numpy() | d.pct.between(0.85, 0.95, inclusive="left").to_numpy()
        d, ordinary, later = d[keep].reset_index(drop=True), ordinary[keep].reset_index(drop=True), later[keep]
    samples = {
        "ordinary researchers (primary)": ordinary.to_numpy(),
        "ordinary, controls not treated within 36 m": (ordinary & ~(d.treated.eq(0) & later)).to_numpy(),
        "all researchers, same matching": np.ones(len(d), bool),
    }
    for v in ("lt5", "5to10", "ge10"):
        samples[f"ordinary, years as last author {v}"] = (ordinary & d.L.eq(v)).to_numpy()
    a_, j_ = pd.factorize(d.author_id)[0], pd.factorize(d.journal_id)[0]
    rng = np.random.default_rng(20261013)
    W = [rng.exponential(1.0, a_.max() + 1)[a_] * rng.exponential(1.0, j_.max() + 1)[j_] for _ in range(B)]
    rows = []
    for lab, m in samples.items():
        s = d[m].reset_index(drop=True)
        c = cells(s, STRATA + MATCH)
        t = s.treated.to_numpy()
        row = {"sample": lab, "treated": int(t.sum()), "controls": int((1 - t).sum()), "authors": int(s.author_id.nunique())}
        est = {}
        for y, nm in (("y_post", "after"), ("y_pre", "before")):
            e, kept = att(c, t, s[y].to_numpy(float), np.ones(len(s)))
            reps = np.array([att(c, t, s[y].to_numpy(float), w[m])[0] for w in W])
            est[nm] = (e, reps)
            row.update(
                {
                    f"att_{nm}": e,
                    f"att_{nm}_lo": np.quantile(reps, 0.025),
                    f"att_{nm}_hi": np.quantile(reps, 0.975),
                    "treated_matched_share": kept,
                    f"rate_treated_{nm}": 1000 * s.loc[t == 1, y].mean(),
                    f"rate_controls_{nm}": 1000 * s.loc[t == 0, y].mean(),
                }
            )
        did = est["after"][1] - est["before"][1]
        row.update({"did": est["after"][0] - est["before"][0], "did_lo": np.quantile(did, 0.025), "did_hi": np.quantile(did, 0.975)})
        rows.append(row)
    save(rows, "first_paper_neartop.csv" if near_top else "first_paper.csv")


def checks():
    """Other windows and design variants."""
    d, ordinary = load()
    ordinary = ordinary.to_numpy()
    near = d.pct.between(0.85, 0.95, inclusive="left").to_numpy()
    top = d.treated.eq(1).to_numpy()
    full = (d.first_publication_year >= 2009).to_numpy()
    variants = {
        "primary": (ordinary, MATCH),
        "highest earlier percentile from years t-5 to t-3": (ordinary, ["career", "prod3", "inst_band", "recent_award", "near_early"]),
        "first OpenAlex paper 2009 or later": (ordinary & full, MATCH),
    }
    ys = {
        "months -24 to <-12": "y_w2412",
        "months -12 to <0": "y_w1200",
        "months -24 to <0": "y_pre",
        "months 0 to <12": "y_w0012",
        "months +12 to <36": "y_post",
        "months 0 to <36": "y_w0036",
    }
    rng = np.random.default_rng(20261018)
    a_, j_ = pd.factorize(d.author_id)[0], pd.factorize(d.journal_id)[0]
    W = [rng.exponential(1.0, a_.max() + 1)[a_] * rng.exponential(1.0, j_.max() + 1)[j_] for _ in range(B)]
    rows = []
    for comp, cm in (("vs other journals", np.ones(len(d), bool)), ("vs near-top journals", top | near)):
        for v, (m0, match) in variants.items():
            m = m0 & cm
            s = d[m].reset_index(drop=True)
            c = cells(s, STRATA + match)
            t = s.treated.to_numpy()
            for k, y in ys.items():
                if v != "primary" and k not in ("months -24 to <0", "months +12 to <36"):
                    continue
                e, kept = att(c, t, s[y].to_numpy(float), np.ones(len(s)))
                reps = np.array([att(c, t, s[y].to_numpy(float), w[m])[0] for w in W])
                rows.append(
                    {
                        "contrast": comp,
                        "variant": v,
                        "window": k,
                        "estimate": e,
                        "lo": np.quantile(reps, 0.025),
                        "hi": np.quantile(reps, 0.975),
                        "treated": int(t.sum()),
                        "controls": int((1 - t).sum()),
                        "treated_matched_share": kept,
                    }
                )
    save(rows, "first_paper_checks.csv")


if __name__ == "__main__":
    run()
    run(near_top=True)
    checks()
