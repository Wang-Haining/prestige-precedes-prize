"""Japan: Japanese-affiliated last authors and KAKEN research grants. Between scientists, directly standardized
differences and risk ratios in the before and after windows and their change; same scientist, order-balanced pairs in
the after window; detection of KAKEN research grants by start year; and the linkage audit.
-> results/japan_between.csv, japan_pairs.csv, japan_detection.csv, japan_audit.csv"""

import numpy as np
import pandas as pd

from analysis.common import DATA, TOP, YS, B, boot_w, pair_est, pairs, save, std_gap

SAMPLES = {"2014-2018": 2018, "2014-2021": 2021}  # index years; 2014-2018 is the main sample
RECORDS = {"all": False, "one researcher number": True}  # all records, or without records linked to >1
DETECT_YEARS = (2012, 2024)


def load_jp():
    """Index papers of Japanese-affiliated last authors, 2014-2021, in file order."""
    d = pd.read_parquet(DATA / "jp_index_papers.parquet")
    pub = pd.to_datetime(d.publication_date)
    d["mo"] = pub.dt.year * 12 + pub.dt.month - 1 + (pub.dt.day - 1) / 31
    d["L"] = "na"  # years as last author are not used for Japan
    return d


def sample(d, last_year, one_nrid):
    """Index papers up to last_year, coded and sorted as in common.load."""
    s = d[d.publication_year.le(last_year) & (d.n_nrid.le(1) if one_nrid else True)].reset_index(drop=True)
    s["s"] = pd.factorize(s.choice_set_id)[0]
    s["a"] = pd.factorize(s.author_id)[0]
    s["j"] = pd.factorize(s.journal_id)[0]
    return s.sort_values(["a", "mo"], kind="mergesort").reset_index(drop=True)


def q(x):
    return np.quantile(x, 0.025), np.quantile(x, 0.975)


def between(d, tag, rec):
    A = (d.pct >= TOP).to_numpy()
    s = d.s.to_numpy()
    Y = {y: d[y].to_numpy(float) for y in YS}
    est = {y: std_gap(s, A, Y[y], np.ones(len(d))) for y in YS}
    rng = np.random.default_rng(20261027)
    reps = {y: [] for y in YS}
    for _ in range(B):
        w = boot_w(d, rng)
        for y in YS:
            reps[y].append(std_gap(s, A, Y[y], w)[:2])
    R = {y: np.array(reps[y]) for y in YS}
    k = s.max() + 1
    both = ((np.bincount(s, A, k) > 0) & (np.bincount(s, ~A, k) > 0))[s]
    common = {
        "sample": tag,
        "records": rec,
        "papers": len(d),
        "scientists": int(d.a.nunique()),
        "papers_top": int(A.sum()),
        "papers_in_sets_with_both": int(both.sum()),
        "papers_top_in_sets_with_both": int((A & both).sum()),
    }
    rows = []
    for y in YS:
        e, r = est[y], R[y]
        rows.append(
            {
                "part": "between",
                "window": y,
                "rd": e[0],
                "rd_lo": q(r[:, 0])[0],
                "rd_hi": q(r[:, 0])[1],
                "rr": e[1],
                "rr_lo": q(r[:, 1])[0],
                "rr_hi": q(r[:, 1])[1],
                "rate_top": e[2],
                "rate_non": e[3],
                **common,
            }
        )
    ch = R["y_post"][:, 0] - R["y_pre"][:, 0]
    lrr = np.log(R["y_post"][:, 1]) - np.log(R["y_pre"][:, 1])
    rows.append(
        {
            "part": "change",
            "window": "after minus before",
            "rd": est["y_post"][0] - est["y_pre"][0],
            "rd_lo": q(ch)[0],
            "rd_hi": q(ch)[1],
            "rr": est["y_post"][1] / est["y_pre"][1],
            "rr_lo": np.exp(q(lrr)[0]),
            "rr_hi": np.exp(q(lrr)[1]),
            **common,
        }
    )
    return rows


def same_scientist(d, tag):
    """Order-balanced pairs of a top-journal paper and another paper at least 24 months apart, after window."""
    rng = np.random.default_rng(20261028)
    P = pairs(d, (d.pct >= TOP).to_numpy().astype(np.int8))
    W = [(rng.exponential(1.0, d.a.max() + 1), rng.exponential(1.0, d.j.max() + 1)) for _ in range(B)]
    e = pair_est(P, d, "y_post")
    reps = pd.DataFrame([r for r in (pair_est(P, d, "y_post", wa, wj) for wa, wj in W) if r is not None])
    row = {"sample": tag, "set": "pairs >=24 m", "window": "y_post", "pairs": len(P), **e, "gap_median_months": float(P.gap.median())}
    for c in ("rd_bal", "rr_bal", "rd_first", "rd_second"):
        row[f"{c}_lo"], row[f"{c}_hi"] = reps[c].quantile(0.025), reps[c].quantile(0.975)
    return [row]


def detection():
    """KAKEN research grants by start year: share with a Latin-script lead investigator name and share reachable by
    the linkage (own name in Latin script, or a researcher number holding a research grant with a Latin-script name)."""
    g = pd.read_parquet(DATA / "jp_kaken_grants.parquet")
    st = pd.to_datetime(g.award_start)
    g["start_year"] = st.dt.year
    g["april1"] = st.dt.month.eq(4) & st.dt.day.eq(1)
    g = g[g.start_year.between(*DETECT_YEARS)]
    rows = []
    for yr, x in g.groupby("start_year"):
        ec = x.early_career
        rows.append(
            {
                "start_year": int(yr),
                "grants": len(x),
                "share_latin_name": x.latin_name.mean(),
                "share_reachable": x.reachable.mean(),
                "early_career_grants": int(ec.sum()),
                "share_reachable_early_career": x.reachable[ec].mean(),
                "other_grants": int((~ec).sum()),
                "share_reachable_other": x.reachable[~ec].mean(),
                "share_april1": x.april1.mean(),
            }
        )
    save(rows, "japan_detection.csv")


def audit():
    """Output corroboration and recall (aggregate input), and the model labels of the audited sample."""
    o = pd.read_csv(DATA / "jp_audit_outputs.csv")
    c = o[o.check.eq("corroboration")].rename(columns={"n": "links_with_outputs", "share": "corroborated"})
    r = o[o.check.eq("recall")].rename(columns={"n": "recall_n", "share": "recall"})
    keys = ["index_years", "window", "paper", "grant_name_script"]
    rows = c[keys + ["links", "links_with_outputs", "corroborated"]].merge(r[keys + ["recall_n", "recall"]], on=keys)
    rows.insert(0, "part", "outputs")
    x = pd.read_csv(DATA / "jp_link_audit.csv")
    m = []
    for by in (["window", "paper"], ["record_nrids"], []):
        for k, g in x.groupby(by) if by else [((), x)]:
            k = k if isinstance(k, tuple) else (k,)
            m.append(
                {
                    "part": "model",
                    **dict(zip(by, k, strict=True)),
                    "model_n": len(g),
                    **{
                        f"model_{v}_share": g.model_label.eq(v).mean() for v in ("same_person", "different_person", "insufficient_evidence")
                    },
                }
            )
    for lab, g in x.groupby("model_label"):
        m.append(
            {
                "part": "model label vs outputs",
                "model_label": lab,
                "model_n": len(g),
                "links_with_outputs": int(g.nrid_has_outputs.sum()),
                "author_on_outputs": int(g.author_on_outputs.sum()),
            }
        )
    out = pd.concat([rows, pd.DataFrame(m)], ignore_index=True)
    for c in ("links", "links_with_outputs", "recall_n", "model_n", "author_on_outputs"):
        out[c] = out[c].astype("Int64")
    save(out, "japan_audit.csv")


if __name__ == "__main__":
    d = load_jp()
    rb, rp = [], []
    for tag, y1 in SAMPLES.items():
        for rec, one in RECORDS.items():
            s = sample(d, y1, one)
            rb += between(s, tag, rec)
            if not one:
                rp += same_scientist(s, tag)
    save(rb, "japan_between.csv")
    save(rp, "japan_pairs.csv")
    detection()
    audit()
