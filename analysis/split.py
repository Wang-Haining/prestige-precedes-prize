"""Between scientists by kind of top journal and domain: the change from the before to the after window for the seven
named journals, the top 5% outside them, and the most cited journal of each topic-year set, overall and by OpenAlex
domain of the paper. Comparator: papers below the top 5% in journals other than the named seven. Cells with fewer than
200 papers in the top group are not estimated. -> results/split_type_domain.csv"""

import numpy as np
import pandas as pd

from analysis.bounds import NAMED
from analysis.common import YS, B, load, save, std_gap

d = load()
named = d.journal_id.isin(list(NAMED)).to_numpy()
other = (d.pct < 0.95).to_numpy() & ~named
groups = {
    "named seven": named,
    "field top 5% (not named)": (d.pct >= 0.95).to_numpy() & ~named,
    "field no. 1 (not named)": d.pct.eq(1.0).to_numpy() & ~named,
    "no. 1, all": d.pct.eq(1.0).to_numpy(),
    "NEJM": d.journal_id.eq("S62468778").to_numpy(),
}
doms = ["all"] + sorted(d.dom_name.dropna().unique())
rows = []
rng = np.random.default_rng(20261028)
for dom in doms:
    D = np.ones(len(d), bool) if dom == "all" else d.dom_name.eq(dom).to_numpy()
    for g, G in groups.items():
        keep = D & (G | other)
        A = G[keep]
        if A.sum() < 200:
            continue
        s = pd.factorize(d.s.to_numpy()[keep])[0]
        aa = pd.factorize(d.a.to_numpy()[keep])[0]
        jj = pd.factorize(d.j.to_numpy()[keep])[0]
        Y = {y: d[y].to_numpy(float)[keep] for y in YS}
        est = {y: std_gap(s, A, Y[y], np.ones(len(A))) for y in YS}
        reps = {y: [] for y in YS}
        for _ in range(B):
            w = rng.exponential(1.0, aa.max() + 1)[aa] * rng.exponential(1.0, jj.max() + 1)[jj]
            for y in YS:
                reps[y].append(std_gap(s, A, Y[y], w)[:2])
        pre, post = np.array(reps["y_pre"]), np.array(reps["y_post"])
        lrr = np.log(post[:, 1]) - np.log(pre[:, 1])
        q = lambda v: (np.quantile(v, 0.025), np.quantile(v, 0.975))  # noqa: E731
        rows.append(
            {
                "domain": dom,
                "group": g,
                "papers_top": int(A.sum()),
                "rate_top_pre": est["y_pre"][2],
                "rate_non_pre": est["y_pre"][3],
                "rate_top_post": est["y_post"][2],
                "rate_non_post": est["y_post"][3],
                "rr_pre": est["y_pre"][1],
                "rr_post": est["y_post"][1],
                "rrr": est["y_post"][1] / est["y_pre"][1],
                "rrr_lo": np.exp(q(lrr)[0]),
                "rrr_hi": np.exp(q(lrr)[1]),
                "change": est["y_post"][0] - est["y_pre"][0],
                "change_lo": q(post[:, 0] - pre[:, 0])[0],
                "change_hi": q(post[:, 0] - pre[:, 0])[1],
            }
        )
save(rows, "split_type_domain.csv")
