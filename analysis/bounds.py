"""Detection-corrected and relative bounds for the three comparisons (top 5% against all other journals), the
first-paper comparison with the broad award linkage, and the named-journal sensitivity analysis.

Detection correction: each group-window rate is divided by the share of paper windows with an acknowledged own award
in which the linkage records a new lead award (false positives taken as negligible). In each bootstrap replicate the
shares are drawn from their binomial distributions given the counts, independently of the replicate weights, which are
those of between.py, same_scientist.py and first_paper.py (same seeds), so the uncorrected rows reproduce the published
intervals. -> results/bounds_detection.csv, bounds_corrected.csv, bounds_relative.csv, bounds_frontier.csv,
bounds_broad.csv, bounds_named.csv"""

import numpy as np
import pandas as pd

from analysis.checks import groups
from analysis.common import DATA, YS, B, att, cells, load, pair_est, pairs, save, std_gap, thin
from analysis.first_paper import MATCH, STRATA
from analysis.first_paper import load as load_cohort

BENCHMARK = 0.10  # relative increase used as the benchmark for the upper limits
FRONTIER = (0, 1, 2, 3, 4, 5)  # percentage points by which first-paper after-window detection is lowered further
# OpenAlex source IDs of the named journals, identified in the 26 June 2026 OpenAlex sources table by ISSN-L
# (one source per ISSN-L): Nature 0028-0836, Science 0036-8075, Cell 0092-8674, New England Journal of Medicine
# 0028-4793, The Lancet 0140-6736, JAMA 0098-7484, Proceedings of the National Academy of Sciences 0027-8424.
NAMED = {
    "S137773608": "Nature",
    "S3880285": "Science",
    "S110447773": "Cell",
    "S62468778": "New England Journal of Medicine",
    "S49861241": "The Lancet",
    "S172573765": "JAMA",
    "S125754415": "Proceedings of the National Academy of Sciences",
}
DET = {  # detection groups (checks.detection) behind each corrected rate
    ("between", "top"): "all papers: top 5%",
    ("between", "other"): "all papers: other journals",
    ("same", "top"): "same-scientist pairs: top-journal paper",
    ("same", "other"): "same-scientist pairs: other paper",
    ("first", "top"): "first top-journal paper (2017-2020)",
    ("first", "other"): "comparison papers, other journals (2017-2020)",
}


def q(x):
    return np.quantile(x, 0.025), np.quantile(x, 0.975)


def link_vars(frame, links):
    """New lead award in the before and after windows, and the funding-history matching variables, from award links."""
    k = (
        frame[["author_id", "publication_year", "publication_date"]]
        .reset_index(names="row")
        .merge(links, on=["author_id", "publication_year"])
    )
    pub, st = k.publication_date, k.award_start

    def months(m):
        return pub + pd.DateOffset(months=m)

    k["y_pre"] = (st >= months(-24)) & (st < pub)
    k["y_post"] = (st >= months(12)) & (st < months(36))
    k["funded_base"] = st < months(-24)
    k["recent_award"] = (st >= months(-60)) & (st < months(-24))
    cols = ["y_pre", "y_post", "funded_base", "recent_award"]
    return k.groupby("row")[cols].max().reindex(range(len(frame)), fill_value=False).astype(int)


def detection_counts(d, broad):
    """Paper windows with an acknowledged own award and those in which the linkage records a new lead award."""
    top, near, ordinary, coh, P = groups(d)
    it, io = np.zeros(len(d), bool), np.zeros(len(d), bool)
    it[P.top.to_numpy()], io[P.non.to_numpy()] = True, True
    gs = {
        "all papers: top 5%": top,
        "all papers: other journals": ~top,
        "same-scientist pairs: top-journal paper": it,
        "same-scientist pairs: other paper": io,
        "first top-journal paper (2017-2020)": coh & ordinary & top,
        "comparison papers, other journals (2017-2020)": coh & ordinary & ~top,
    }
    yb = link_vars(d, broad)
    rows = []
    for g, m in gs.items():
        for w, rc, yc in (("before window", "ref_pre", "y_pre"), ("after window", "ref_post", "y_post")):
            r = m & (d[rc].to_numpy() == 1)
            for lk, y in (("main", d[yc].to_numpy()), ("broad", yb[yc].to_numpy())):
                if lk == "broad" and not g.startswith(("first", "comparison")):
                    continue
                rows.append(
                    {
                        "linkage": lk,
                        "group": g,
                        "window": w,
                        "windows": int(r.sum()),
                        "detected": int(y[r].sum()),
                        "share": float(y[r].mean()),
                    }
                )
    return pd.DataFrame(rows)


def share_draws(det):
    """Detection share and its binomial bootstrap draws for each group and window (main linkage)."""
    rng = np.random.default_rng(20261027)
    out = {}
    for (comp, arm), g in DET.items():
        for w, yc in (("before window", "y_pre"), ("after window", "y_post")):
            r = det[(det.linkage == "main") & (det.group == g) & (det.window == w)].iloc[0]
            out[(comp, arm, yc)] = (r.detected / r.windows, rng.binomial(r.windows, r.detected / r.windows, B) / r.windows)
    return out


def div(S, comp, y, corr):
    """Detection shares dividing the (top, other) rates: point estimates and replicate draws."""
    if corr == "none":
        return (1.0, 1.0), (1.0, 1.0)
    st, so = S[(comp, "top", y)], S[(comp, "other", y)]
    return (st[0], so[0]), (st[1], so[1])


def row(comp, measure, corr, est, reps, **kw):
    lo, hi = q(reps)
    return {"comparison": comp, "measure": measure, "correction": corr, "estimate": est, "lo": lo, "hi": hi, **kw}


def between(d, S, named):
    """Between scientists, standardized over topic-year sets, with the replicate weights of between.py."""
    s = d.s.to_numpy()
    A = {"top 5%": (d.pct >= 0.95).to_numpy(), "named journals": named}
    Y = {y: d[y].to_numpy(float) for y in YS}
    est = {(k, y): std_gap(s, a, Y[y], np.ones(len(d))) for k, a in A.items() for y in YS}
    rng = np.random.default_rng(20261010)
    aa, jj = d.a.to_numpy(), d.j.to_numpy()
    reps = {k: [] for k in est}
    for _ in range(B):
        w = rng.exponential(1.0, aa.max() + 1)[aa] * rng.exponential(1.0, jj.max() + 1)[jj]
        for k, a in A.items():
            for y in YS:
                reps[(k, y)].append(std_gap(s, a, Y[y], w))
    R = {k: np.array(v) for k, v in reps.items()}
    E = {k: np.array(v) for k, v in est.items()}
    rows, rel = [], []
    for corr in ("none", "detection"):
        out = {}
        for y in YS:
            (pt_, po_), (rt_, ro_) = div(S, "between", y, corr)
            pt, pc = E[("top 5%", y)][2] / pt_, E[("top 5%", y)][3] / po_
            rt, rc = R[("top 5%", y)][:, 2] / rt_, R[("top 5%", y)][:, 3] / ro_
            out[y] = (pt, pc, rt, rc)
            win = "before" if y == "y_pre" else "after"
            rows.append(row("between scientists", f"rate, top 5%, {win} window", corr, pt, rt))
            rows.append(row("between scientists", f"rate, other journals, {win} window", corr, pc, rc))
            rows.append(row("between scientists", f"difference, {win} window", corr, pt - pc, rt - rc))
            rows.append(row("between scientists", f"risk ratio, {win} window", corr, pt / pc, rt / rc))
        (bt, bc, rbt, rbc), (at, ac, rat, rac) = out["y_pre"], out["y_post"]
        rows.append(row("between scientists", "change, after minus before", corr, (at - ac) - (bt - bc), (rat - rac) - (rbt - rbc)))
        rr, rrr = (at / ac) / (bt / bc), (rat / rac) / (rbt / rbc)
        rows.append(row("between scientists", "ratio of risk ratios, after / before", corr, rr, rrr))
        rel.append(relative("between scientists", "ratio of risk ratios minus 1", corr, rr - 1, rrr - 1))
    # named journals (uncorrected)
    named_rows = []
    for y in YS:
        e, r = E[("named journals", y)], R[("named journals", y)]
        named_rows.append(
            {
                "comparison": "between scientists",
                "window": y,
                "rd": e[0],
                "rd_lo": q(r[:, 0])[0],
                "rd_hi": q(r[:, 0])[1],
                "rr": e[1],
                "rr_lo": q(r[:, 1])[0],
                "rr_hi": q(r[:, 1])[1],
                "rate_top": e[2],
                "rate_other": e[3],
                "papers_top": int(named.sum()),
                "papers_other": int((~named).sum()),
            }
        )
    e1, e0 = E[("named journals", "y_post")], E[("named journals", "y_pre")]
    r1, r0 = R[("named journals", "y_post")], R[("named journals", "y_pre")]
    lo, hi = q(r1[:, 0] - r0[:, 0])
    llo, lhi = q(np.log(r1[:, 1]) - np.log(r0[:, 1]))
    named_rows.append(
        {
            "comparison": "between scientists",
            "window": "after minus before",
            "rd": e1[0] - e0[0],
            "rd_lo": lo,
            "rd_hi": hi,
            "rr": e1[1] / e0[1],
            "rr_lo": np.exp(llo),
            "rr_hi": np.exp(lhi),
        }
    )
    return rows, rel, named_rows


def relative(comp, measure, corr, est, reps, over_rate=None):
    """Relative measure with its upper 95% limit from the replicates of the ratio; over_rate is the upper limit of the
    absolute measure divided by the point estimate of the comparison rate (for ratio measures, the upper limit)."""
    hi = q(reps)[1]
    return {
        "comparison": comp,
        "measure": measure,
        "correction": corr,
        "estimate": est,
        "upper_limit": hi,
        "upper_limit_over_rate": hi if over_rate is None else over_rate,
        "benchmark": BENCHMARK,
        "below_benchmark": bool(hi < BENCHMARK),
    }


def pair_weights(d):
    """Replicate weights of same_scientist.run (all fields), regenerated from its seed: the draws for the between- and
    within-scientist curves come first and are skipped."""
    rng = np.random.default_rng(20261011)
    t = thin(d)
    for frame, n in ((d, 2), (t, 1)):
        na, nj = frame.a.max() + 1, frame.j.max() + 1
        for _ in range(n * B):
            rng.exponential(1.0, na)
            rng.exponential(1.0, nj)
    return rng, d.a.max() + 1, d.j.max() + 1


def same_scientist(d, S, named):
    P = {"top 5%": pairs(d, (d.pct >= 0.95).to_numpy().astype(np.int8)), "named journals": pairs(d, named.astype(np.int8))}
    est = {k: pair_est(p, d, "y_post") for k, p in P.items()}
    rng, na, nj = pair_weights(d)
    reps = {k: [] for k in P}
    for _ in range(B):
        wa, wj = rng.exponential(1.0, na), rng.exponential(1.0, nj)
        for k, p in P.items():
            reps[k].append(pair_est(p, d, "y_post", wa, wj))
    reps = {k: pd.DataFrame(v) for k, v in reps.items()}
    e, r = est["top 5%"], reps["top 5%"]
    rows, rel = [], []
    for corr in ("none", "detection"):
        (pt, po), (rt, ro) = div(S, "same", "y_post", corr)
        rows.append(row("same scientist", "rate, top-journal paper, after window", corr, e["rate_top"] / pt, r.rate_top.to_numpy() / rt))
        rows.append(row("same scientist", "rate, other paper, after window", corr, e["rate_non"] / po, r.rate_non.to_numpy() / ro))
        rows.append(
            row(
                "same scientist",
                "difference, after window",
                corr,
                e["rate_top"] / pt - e["rate_non"] / po,
                r.rate_top.to_numpy() / rt - r.rate_non.to_numpy() / ro,
                pairs=len(P["top 5%"]),
            )
        )
        rr, rrr = e["rr_bal"] * po / pt, r.rr_bal.to_numpy() * ro / rt
        rows.append(row("same scientist", "risk ratio, after window", corr, rr, rrr, pairs=len(P["top 5%"])))
        rel.append(relative("same scientist", "risk ratio minus 1", corr, rr - 1, rrr - 1))
    e, r = est["named journals"], reps["named journals"]
    named_row = {
        "comparison": "same scientist",
        "window": "y_post",
        "rd": e["rd_bal"],
        "rd_lo": r.rd_bal.quantile(0.025),
        "rd_hi": r.rd_bal.quantile(0.975),
        "rr": e["rr_bal"],
        "rr_lo": r.rr_bal.quantile(0.025),
        "rr_hi": r.rr_bal.quantile(0.975),
        "rate_top": e["rate_top"],
        "rate_other": e["rate_non"],
        "pairs": len(P["named journals"]),
    }
    return rows, rel, named_row


def att_parts(c, t, y, w):
    """Rate of the first top-journal papers and weighted rate of their matched comparison papers (per 1,000), over
    cells with both kinds of paper."""
    k = c.max() + 1
    wt, wc = w * t, w * (1 - t)
    nt, nc = np.bincount(c, wt, k), np.bincount(c, wc, k)
    st, sc = np.bincount(c, wt * y, k), np.bincount(c, wc * y, k)
    ok = (nt > 0) & (nc > 0)
    return 1000 * st[ok].sum() / nt[ok].sum(), 1000 * (nt[ok] * sc[ok] / nc[ok]).sum() / nt[ok].sum()


def first_paper(S, broad):
    """First top-journal paper, with the replicate weights of first_paper.run: main linkage (uncorrected and
    detection-corrected), broad linkage, and named journals."""
    d, ordinary = load_cohort()
    lb = link_vars(d, broad)
    nd = pd.read_parquet(DATA / "named_paper_dates.parquet")
    first_named = pd.to_datetime(d.author_id.map(nd.groupby("author_id").publication_date.min()))
    ordinary_named = (first_named.isna() | (first_named >= d.publication_date)).to_numpy()
    named = d.journal_id.isin(NAMED).to_numpy().astype(int)
    db = d.copy()
    db["funded_base"], db["recent_award"] = lb.funded_base.astype(bool).to_numpy(), lb.recent_award.astype(float).to_numpy()
    for y in YS:
        db[y] = lb[y].to_numpy()
    variants = {  # sample mask, frame, treatment
        "main": (ordinary.to_numpy(), d, d.treated.to_numpy()),
        "broad": (ordinary.to_numpy(), db, d.treated.to_numpy()),
        "named": (ordinary_named, d, named),
    }
    prep = {}
    for k, (m, f, t) in variants.items():
        s = f[m].reset_index(drop=True)
        prep[k] = (m, cells(s, STRATA + MATCH), t[m], {y: s[y].to_numpy(float) for y in YS}, s)
    est = {(k, y): att_parts(c, t, Y[y], np.ones(len(t))) for k, (_m, c, t, Y, _s) in prep.items() for y in YS}
    for y in YS:  # the parts reproduce the matched difference of first_paper.py
        m, c, t, Y, s = prep["main"]
        assert abs(est[("main", y)][0] - est[("main", y)][1] - att(c, t, Y[y], np.ones(len(t)))[0]) < 1e-9
    a_, j_ = pd.factorize(d.author_id)[0], pd.factorize(d.journal_id)[0]
    rng = np.random.default_rng(20261013)
    reps = {k: [] for k in est}
    for _ in range(B):
        w = rng.exponential(1.0, a_.max() + 1)[a_] * rng.exponential(1.0, j_.max() + 1)[j_]
        for k, (m, c, t, Y, _s) in prep.items():
            for y in YS:
                reps[(k, y)].append(att_parts(c, t, Y[y], w[m]))
    E = {k: np.array(v) for k, v in est.items()}
    R = {k: np.array(v) for k, v in reps.items()}

    def parts(k, corr, delta=0.0):
        """(treated, comparison) rates in the before and after windows: point estimates and replicates."""
        out = {}
        for y in YS:
            e, r = E[(k, y)], R[(k, y)]
            if corr == "detection":
                st, so = S[("first", "top", y)], S[("first", "other", y)]
                dlt = delta if y == "y_post" else 0.0
                e = e / np.array([st[0] - dlt, so[0]])
                r = r / np.column_stack([st[1] - dlt, so[1]])
            out[y] = (e, r)
        return out

    rows, rel = [], []
    comp = "first top-journal paper"
    for corr in ("none", "detection"):
        p = parts("main", corr)
        for y in YS:
            win = "before" if y == "y_pre" else "after"
            (e, r) = p[y]
            rows.append(row(comp, f"rate, first top-journal papers, {win} window", corr, e[0], r[:, 0]))
            rows.append(row(comp, f"rate, matched comparison papers, {win} window", corr, e[1], r[:, 1]))
            rows.append(row(comp, f"difference, {win} window", corr, e[0] - e[1], r[:, 0] - r[:, 1]))
            rows.append(row(comp, f"risk ratio, {win} window", corr, e[0] / e[1], r[:, 0] / r[:, 1]))
        (eb, rb), (ea, ra) = p["y_pre"], p["y_post"]
        did, rdid = (ea[0] - ea[1]) - (eb[0] - eb[1]), (ra[:, 0] - ra[:, 1]) - (rb[:, 0] - rb[:, 1])
        rows.append(row(comp, "change, after minus before", corr, did, rdid))
        rrr, rrrr = (ea[0] / ea[1]) / (eb[0] / eb[1]), (ra[:, 0] / ra[:, 1]) / (rb[:, 0] / rb[:, 1])
        rows.append(row(comp, "ratio of risk ratios, after / before", corr, rrr, rrrr))
        rel.append(relative(comp, "change / matched comparison rate, after window", corr, did / ea[1], rdid / ra[:, 1], q(rdid)[1] / ea[1]))
        rel.append(
            relative(
                comp,
                "after-window difference / matched comparison rate",
                corr,
                ea[0] / ea[1] - 1,
                ra[:, 0] / ra[:, 1] - 1,
                q(ra[:, 0] - ra[:, 1])[1] / ea[1],
            )
        )
        rel.append(relative(comp, "ratio of risk ratios minus 1", corr, rrr - 1, rrrr - 1))
    frontier = []
    for pp in FRONTIER:
        p = parts("main", "detection", pp / 100)
        (eb, rb), (ea, ra) = p["y_pre"], p["y_post"]
        did, rdid = (ea[0] - ea[1]) - (eb[0] - eb[1]), (ra[:, 0] - ra[:, 1]) - (rb[:, 0] - rb[:, 1])
        lo, hi = q(rdid)
        frontier.append(
            {
                "detection_lowered_pp": pp,
                "detection_first_after": S[("first", "top", "y_post")][0] - pp / 100,
                "detection_comparison_after": S[("first", "other", "y_post")][0],
                "change": did,
                "change_lo": lo,
                "change_hi": hi,
                "comparison_rate_after": ea[1],
                "relative_change": did / ea[1],
                "relative_change_hi": q(rdid / ra[:, 1])[1],
            }
        )
    # broad linkage (uncorrected), in the layout of first_paper.csv
    m, c, t, Y, s = prep["broad"]
    broad_row = {
        "sample": "ordinary researchers (primary)",
        "treated": int(t.sum()),
        "controls": int((1 - t).sum()),
        "authors": int(s.author_id.nunique()),
        "treated_matched_share": att(c, t, Y["y_post"], np.ones(len(t)))[1],
    }
    for y, nm in (("y_post", "after"), ("y_pre", "before")):
        e, r = E[("broad", y)], R[("broad", y)]
        lo, hi = q(r[:, 0] - r[:, 1])
        broad_row.update(
            {f"att_{nm}": e[0] - e[1], f"att_{nm}_lo": lo, f"att_{nm}_hi": hi, f"rate_treated_{nm}": e[0], f"rate_matched_{nm}": e[1]}
        )
    ra, rb = R[("broad", "y_post")], R[("broad", "y_pre")]
    rdid = (ra[:, 0] - ra[:, 1]) - (rb[:, 0] - rb[:, 1])
    broad_row["did"] = broad_row["att_after"] - broad_row["att_before"]
    broad_row["did_lo"], broad_row["did_hi"] = q(rdid)
    broad_row["did_relative_hi"] = q(rdid / ra[:, 1])[1]
    # named journals (uncorrected), in the layout of bounds_named.csv
    m, c, t, Y, s = prep["named"]
    named_rows = []
    for y in YS:
        e, r = E[("named", y)], R[("named", y)]
        named_rows.append(
            {
                "comparison": comp,
                "window": y,
                "rd": e[0] - e[1],
                "rd_lo": q(r[:, 0] - r[:, 1])[0],
                "rd_hi": q(r[:, 0] - r[:, 1])[1],
                "rr": e[0] / e[1],
                "rr_lo": q(r[:, 0] / r[:, 1])[0],
                "rr_hi": q(r[:, 0] / r[:, 1])[1],
                "rate_top": e[0],
                "rate_other": e[1],
                "papers_top": int(t.sum()),
                "papers_other": int((1 - t).sum()),
                "treated_matched_share": att(c, t, Y[y], np.ones(len(t)))[1],
            }
        )
    (eb, rb), (ea, ra) = (E[("named", "y_pre")], R[("named", "y_pre")]), (E[("named", "y_post")], R[("named", "y_post")])
    rdid = (ra[:, 0] - ra[:, 1]) - (rb[:, 0] - rb[:, 1])
    lrr = np.log(ra[:, 0] / ra[:, 1]) - np.log(rb[:, 0] / rb[:, 1])
    named_rows.append(
        {
            "comparison": comp,
            "window": "after minus before",
            "rd": (ea[0] - ea[1]) - (eb[0] - eb[1]),
            "rd_lo": q(rdid)[0],
            "rd_hi": q(rdid)[1],
            "rr": (ea[0] / ea[1]) / (eb[0] / eb[1]),
            "rr_lo": np.exp(q(lrr)[0]),
            "rr_hi": np.exp(q(lrr)[1]),
        }
    )
    return rows, rel, frontier, broad_row, named_rows


if __name__ == "__main__":
    d = load()
    broad = pd.read_parquet(DATA / "award_links_broad.parquet")
    det = detection_counts(d, broad)
    save(det, "bounds_detection.csv")
    S = share_draws(det)
    named = d.journal_id.isin(NAMED).to_numpy()
    rows_b, rel_b, named_b = between(d, S, named)
    rows_s, rel_s, named_s = same_scientist(d, S, named)
    rows_f, rel_f, frontier, broad_row, named_f = first_paper(S, broad)
    save(rows_b + rows_s + rows_f, "bounds_corrected.csv")
    save(rel_b + rel_s + rel_f, "bounds_relative.csv")
    save(frontier, "bounds_frontier.csv")
    save([broad_row], "bounds_broad.csv")
    save(named_b + [named_s] + named_f, "bounds_named.csv")
