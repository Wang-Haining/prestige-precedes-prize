#!/usr/bin/env python3
"""Supporting Tables S3-S12 from the aggregate results. python -m analysis.tables [RESULTS_DIR] [OUT_TEX]"""

import sys
import textwrap
from pathlib import Path

import pandas as pd

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "tables" / "supporting_tables.tex"
DEFS = [("top 5%", "Top 5\\%"), ("top decile", "Top 10\\%"), ("top quartile", "Top 25\\%"), ("top journal in set", "Most cited journal")]
FIRST_LABELS = {
    "ordinary researchers (primary)": "No earlier top-journal paper",
    "ordinary, controls not treated within 36 m": "Comparison researchers without a top-journal paper in the next 36 months",
    "all researchers, same matching": "All researchers (matched between scientists)",
    "ordinary, years as last author lt5": "No earlier top-journal paper, $<5$ years as last author",
    "ordinary, years as last author 5to10": "No earlier top-journal paper, 5 to $<10$ years",
    "ordinary, years as last author ge10": "No earlier top-journal paper, $\\geq 10$ years",
}


def n(x, d=1):
    s = f"{x:,.{d}f}"
    return s.replace("-", "$-$") if x < 0 else s


def ci(e, lo, hi, d=1):
    return f"{n(e, d)} ({n(lo, d)} to {n(hi, d)})"


def months(w):
    return w.replace("months ", "").replace("-", "$-$").replace("<", "$<$")


def stack(top, bottom):
    """A two-line numeric cell (estimate over interval) for tables with many interval columns."""
    return f"\\makecell[rt]{{{top}\\\\{bottom}}}"


def header(text, a):
    """Wrap a long column header onto short lines."""
    if len(text) <= 13:
        return text
    lines = textwrap.wrap(text, 13, break_long_words=False)
    return f"\\makecell[{a}b]{{{'\\\\'.join(lines)}}}"


def table(label, caption, cols, rows, align=None):
    """rows: list of cell lists; a str row is a panel heading spanning all columns. All tables share one type size;
    the X column (by default the first) wraps to fit the text width."""
    align = align or "l" + "r" * (len(cols) - 1)
    if "X" not in align:
        align = "X" + align[1:]
    spec = align.replace("X", ">{\\raggedright\\arraybackslash}X")
    heads = [c if a == "X" else header(c, a) for c, a in zip(cols, align, strict=True)]
    out = [
        f"\\begin{{table}}[H]\n\\centering\\small\n\\caption{{{caption}}}\\label{{{label}}}\n\\footnotesize",
        f"\\begin{{tabularx}}{{\\linewidth}}{{{spec}}}\n\\toprule",
        " & ".join(heads) + " \\\\\n\\midrule",
    ]
    for r in rows:
        out.append(
            f"\\multicolumn{{{len(cols)}}}{{>{{\\raggedright\\arraybackslash}}p{{\\dimexpr\\textwidth-2\\tabcolsep\\relax}}}}{{\\textit{{{r}}}}} \\\\"
            if isinstance(r, str)
            else " & ".join(r) + " \\\\"
        )
    out.append("\\bottomrule\n\\end{tabularx}\n\\end{table}\n")
    return "\n".join(out)


def linkage():
    rec = pd.read_csv(SRC / "recall_by_group.csv")
    det = pd.read_csv(SRC / "detection.csv")
    esc = lambda s: (
        (s[:1].upper() + s[1:]).replace("%", "\\%").replace("2017-2020", "2017--2020").replace("<", "$<$").replace(">=", "$\\geq$")
    )
    rows = ["Award-level recall: share of acknowledged awards led by the author, months $-24$ to $<36$"]
    rows += [[esc(x.group), f"{int(x.reference_awards):,}", f"{100 * x.recall:.1f}", ""] for _, x in rec.iterrows()]
    rows.append("Detection of the binary outcome: share of windows with an acknowledged award in which a new lead award is recorded")
    piv = det.pivot_table(index="group", columns="window", values=["paper_windows_with_reference", "detected"], sort=False)
    for g, r in piv.iterrows():
        rows.append(
            [
                esc(g),
                f"{int(r[('paper_windows_with_reference', 'before window')]):,} / {int(r[('paper_windows_with_reference', 'after window')]):,}",
                f"{100 * r[('detected', 'before window')]:.1f}",
                f"{100 * r[('detected', 'after window')]:.1f}",
            ]
        )
    return table(
        "tab:s_linkage_groups",
        "Award linkage by analysis group. In the recall rows, the columns give reference awards and recall (\\%). In the detection rows, they give paper windows with a reference award (before / after) and the share detected (\\%) in each window.",
        ["Group", "Reference awards or windows", "Recall or detected, before (\\%)", "Detected, after (\\%)"],
        rows,
        align="lrrr",
    )


def between():
    s = pd.read_csv(SRC / "between.csv")
    b = s[s.part.eq("between")]
    rows = ["Top journals against all other journals"]
    for k, lab in DEFS:
        a = b[(b.label == k) & (b.window == "y_post")].iloc[0]
        p = b[(b.label == k) & (b.window == "y_pre")].iloc[0]
        rows.append(
            [
                lab,
                f"{int(a.papers_top):,}",
                f"{n(a.rate_top)} / {n(a.rate_non)}",
                ci(a.rd, a.rd_lo, a.rd_hi),
                ci(a.rr, a.rr_lo, a.rr_hi, 2),
                ci(p.rd, p.rd_lo, p.rd_hi),
                n(a.rd / 4, 2),
            ]
        )
    sn = pd.read_csv(SRC / "between_neartop.csv")
    deltas = sn[sn.part == "delta"].delta_per_1000.tolist()
    btw = sn[sn.part == "between"].reset_index(drop=True)
    rows.append("Top 5\\% against near-top journals")
    for i, lab in enumerate(["Percentile 0.85 to $<$0.95", "Percentile 0.90 to $<$0.95"]):
        a, p = btw.iloc[2 * i + 1], btw.iloc[2 * i]
        rows.append(
            [
                lab,
                f"{int(a.papers_top):,} / {int(a.papers - a.papers_top):,}",
                f"{n(a.rate_top)} / {n(a.rate_non)}",
                ci(a.rd, a.rd_lo, a.rd_hi),
                ci(a.rr, a.rr_lo, a.rr_hi, 2),
                ci(p.rd, p.rd_lo, p.rd_hi),
                n(deltas[i], 2),
            ]
        )
    return table(
        "tab:s_between",
        "Between scientists: new lead awards per 1,000 index papers after top-journal papers and after comparison papers, directly standardized over topic-year sets, by definition of a top journal and against near-top journals. In the near-top rows, both rates are standardized over the topic-year sets that contain both kinds of journal. The smallest difference of interest ($\\delta$) is one quarter of the after-window difference.",
        [
            "Comparison",
            "Top-journal papers (/ near-top papers)",
            "Rates, after (top / comparison)",
            "After: difference",
            "After: risk ratio",
            "Before: difference",
            "$\\delta$",
        ],
        rows,
    )


def timing():
    r = pd.concat([pd.read_csv(SRC / "timing.csv"), pd.read_csv(SRC / "cumulative.csv")], ignore_index=True)
    r = r[~r.window.isin(["months -24 to <0", "months +12 to <36"])]
    rows = [
        [
            x.comparison.replace(">=", "$\\geq$"),
            x.contrast.replace("vs ", ""),
            months(x.window),
            ci(x.estimate, x.lo, x.hi),
            f"{n(x.rate_top)} / {n(x.rate_other)}",
        ]
        for _, x in r.iterrows()
    ]
    return table(
        "tab:s_timing",
        "Differences in new lead awards per 1,000 index papers, top 5\\% of journals minus other or near-top journals, in windows other than the before and after windows (Tables~S4 and~S8 give those). Same-scientist pairs are far enough apart that the windows compared do not overlap.",
        ["Comparison", "Comparator", "Window (months)", "Difference (95\\% CI)", "Rates, top / comparator"],
        rows,
        align="lllrr",
    )


def curves():
    c = pd.read_csv(SRC / "curves.csv")
    c = c[c.field.eq("all")]
    piv = {}
    for (comp, w), g in c.groupby(["comparison", "window"]):
        for _, r in g.iterrows():
            piv.setdefault((r.pct_lo, r.pct_hi), {})[(comp, w)] = ci(r.rd_vs_top, r.lo, r.hi)
    rows = [
        [f"{lo:.2f}--{hi:.2f}", v.get(("between", "y_pre"), ""), v.get(("between", "y_post"), ""), v.get(("within", "y_post"), "")]
        for (lo, hi), v in sorted(piv.items())
    ]
    h = pd.read_csv(SRC / "hinge.csv")
    h = h[h.field.eq("all")]
    rows2 = []
    for _, r in h.iterrows():
        nm = {"between": "Between scientists", "within": "Same scientist"}[r.comparison] + (
            ", before" if r.window == "y_pre" else ", after"
        )
        rows2.append(
            [
                nm,
                ci(r.knot, r.knot_lo, r.knot_hi, 3),
                ci(r.slope_below, r.slope_below_lo, r.slope_below_hi, 2),
                ci(r.slope_above, r.slope_above_lo, r.slope_above_hi, 2),
                ci(r.slope_linear, r.slope_linear_lo, r.slope_linear_hi, 2),
            ]
        )
    return table(
        "tab:s_curves",
        "New lead awards per 1,000 index papers by band of the percentile of citation standing, difference from the top band (0.95--1.00) in a linear probability model. Between scientists: topic-year set fixed effects, all index papers. Same scientist: scientist fixed effects and index year, index papers at least 24 months apart.",
        ["Percentile band", "Between, before", "Between, after", "Same scientist, after"],
        rows,
    ) + table(
        "tab:s_hinge",
        "Fits of new lead awards on the percentile of citation standing, with the fixed effects of Table~S6: a straight line, and a line with one break point (knot) chosen by best fit. Slopes per 1,000 index papers for each tenth of the percentile range. Intervals are from 200 bootstrap replicates.",
        ["Comparison", "Knot", "Slope below", "Slope above", "Straight-line slope"],
        rows2,
    )


SETS = [
    ("pairs >=24 m", "All pairs (primary)"),
    ("pairs >=24 m, L=lt5", "Years as last author $<5$"),
    ("pairs >=24 m, L=5to10", "Years as last author 5 to $<10$"),
    ("pairs >=24 m, L=ge10", "Years as last author $\\geq 10$"),
    ("difference: L<5 minus L>=10", "Difference, $<5$ minus $\\geq 10$ years"),
    ("pairs >=24 m, F=0", "No award started in the 60 months before the earlier paper"),
    ("pairs >=24 m, F=1", "Award started in the 60 months before the earlier paper"),
    ("pairs >=24 m, index years <=2020", "Index years to 2020"),
    ("before window, pairs >=60 m", "Before window, pairs $\\geq 60$ months apart"),
    ("pairs <24 m (windows overlap)", "Pairs $<24$ months apart (windows overlap)"),
    ("fixed effects, papers >=24 m apart", "Fixed effects, papers $\\geq 24$ months apart"),
    ("pairs >=24 m, scientist-only bootstrap", "All pairs, bootstrap of scientists only"),
]


def pair_row(r, name, key=""):
    order = "" if pd.isna(r.get("rd_first")) else f"{n(r.rd_first)} / {n(r.rd_second)}"
    npairs = "" if pd.isna(r.get("n_first")) or key.startswith(("fixed", "difference")) else f"{int(r.n_first):,} / {int(r.n_second):,}"
    rr = "" if pd.isna(r.get("rr_bal")) or pd.isna(r.get("rr_bal_lo")) else ci(r.rr_bal, r.rr_bal_lo, r.rr_bal_hi, 2)
    return [name, npairs, ci(r.rd_bal, r.rd_bal_lo, r.rd_bal_hi), rr, order]


def pairs():
    p = pd.read_csv(SRC / "pairs.csv")
    p = p[p.field.eq("all")]
    ab = pd.read_csv(SRC / "pairs_scientist_boot.csv")
    allp = pd.concat([p, ab.assign(field="all")], ignore_index=True)
    get = lambda d, s: allp[(allp.definition == d) & (allp.set == s)]
    rows = ["Top 5\\% against all other journals"]
    for key, name in SETS:
        r = get("top 5%", key)
        if not r.empty:
            rows.append(pair_row(r.iloc[0], name, key))
    g = pd.read_csv(SRC / "pair_checks.csv")
    for _, x in g[(g.part == "stratum") & (g.contrast == "vs other journals")].iterrows():
        rows.append(
            [
                x.stratum.replace("months apart", "Pairs").replace("<", "$<$").replace(">=", "$\\geq$") + " months apart",
                f"{int(x.n_first):,} / {int(x.n_second):,}",
                ci(x.estimate, x.lo, x.hi),
                "",
                "",
            ]
        )
    so = pd.read_csv(SRC / "standardized_order.csv")
    for _, x in so.iterrows():
        if x.contrast == "vs other journals":
            rows.append(
                [
                    "Standardized by the years of both papers, field, years as last author, and prior funding",
                    "",
                    ci(x.estimate, x.lo, x.hi),
                    "",
                    "",
                ]
            )
    rows.append("Other definitions of a top journal, against all other journals")
    for k, lab in DEFS[1:]:
        rows.append(pair_row(get(k, "pairs >=24 m").iloc[0], lab))
    rows.append("Top 5\\% against near-top journals")
    pn = pd.read_csv(SRC / "pairs_neartop.csv")
    for k, name in SETS[:6] + [SETS[8]]:
        r = pn[(pn.definition == "near-top 0.85-0.95") & (pn.set == k)]
        if not r.empty:
            rows.append(pair_row(r.iloc[0], name, k))
    for _, x in g[(g.part == "stratum") & (g.contrast == "vs near-top journals")].iterrows():
        rows.append(
            [
                x.stratum.replace("months apart", "Pairs").replace("<", "$<$").replace(">=", "$\\geq$") + " months apart",
                f"{int(x.n_first):,} / {int(x.n_second):,}",
                ci(x.estimate, x.lo, x.hi),
                "",
                "",
            ]
        )
    for _, x in so.iterrows():
        if x.contrast == "vs near-top journals":
            rows.append(
                [
                    "Standardized by the years of both papers, field, years as last author, and prior funding",
                    "",
                    ci(x.estimate, x.lo, x.hi),
                    "",
                    "",
                ]
            )
    r = pn[(pn.definition == "near-top 0.90-0.95") & (pn.set == "pairs >=24 m")]
    if not r.empty:
        rows.append(pair_row(r.iloc[0], "Near-top band 0.90 to $<$0.95"))
    r0 = get("top 5%", "pairs >=24 m").iloc[0]
    rates = {}
    for o in ("first", "second"):
        non = r0[f"rd_{o}"] / (r0[f"rr_{o}"] - 1)
        rates[o] = (non * r0[f"rr_{o}"], non)
    note = (
        f" For the top 5\\%, award rates after the earlier and the later paper were {n(rates['first'][0])} and {n(rates['first'][1])} when the top-journal paper came first,"
        f" and {n(rates['second'][1])} and {n(rates['second'][0])} when it came second."
        f" Averaged over the two orders, the rates were {n((rates['first'][0] + rates['second'][0]) / 2)} after the top-journal paper and {n((rates['first'][1] + rates['second'][1]) / 2)} after the other paper."
    )
    return table(
        "tab:s_pairs",
        "Same scientist: new lead awards per 1,000 index papers after the top-journal paper minus after the other paper, averaged over the two orders, in the after window unless stated. Pairs are listed as top journal first / top journal second, and the last column gives the difference within each order. Years as last author and awards started before the pair are measured at the earlier paper of the pair."
        + note,
        ["Sample", "Pairs (first / second)", "Difference (95\\% CI)", "Risk ratio", "By order (first / second)"],
        rows,
    )


def profile():
    r = pd.read_csv(SRC / "pair_checks.csv")
    pr = r[r.part == "profile"]
    rows = [
        [
            x.contrast.replace("vs ", ""),
            x.stratum,
            f"{int(x.pairs):,}",
            f"{100 * x.share_lt5:.1f}",
            f"{100 * x.share_ge10:.1f}",
            f"{100 * x.share_award_60m:.1f}",
            f"{x.months_apart_median:.1f}",
            f"{x.first_year_mean:.1f}",
        ]
        for _, x in pr.iterrows()
    ]
    return table(
        "tab:s_pairs_profile",
        "Same scientist: description of the pairs in each order, measured at the earlier paper of the pair.",
        [
            "Comparator",
            "Order",
            "Pairs",
            "$<$5 years as last author (\\%)",
            "$\\geq$10 years (\\%)",
            "Award started in prior 60 months (\\%)",
            "Median months apart",
            "Mean year of earlier paper",
        ],
        rows,
        align="llrrrrrr",
    )


def fields():
    rows = []
    for d, lab in (
        ("health", "Health Sciences"),
        ("life", "Life Sciences"),
        ("physical", "Physical Sciences"),
        ("social", "Social Sciences"),
    ):
        p = pd.read_csv(SRC / f"pairs_{d}.csv")
        h = pd.read_csv(SRC / f"hinge_{d}.csv")
        r = p[p.definition.eq("top 5%")].iloc[0]
        hb = h[(h.comparison == "between") & (h.window == "y_post")].iloc[0]
        hw = h[h.comparison == "within"].iloc[0]
        rows.append(
            [
                lab,
                f"{int(r.n_first + r.n_second):,}",
                ci(r.rd_bal, r.rd_bal_lo, r.rd_bal_hi),
                ci(hb.slope_linear, hb.slope_linear_lo, hb.slope_linear_hi, 2),
                ci(hw.slope_linear, hw.slope_linear_lo, hw.slope_linear_hi, 2),
            ]
        )
    return table(
        "tab:s_fields",
        "By field (OpenAlex domain of the paper's primary topic): same-scientist difference for the top 5\\% in the after window (per 1,000 index papers), and straight-line slopes per tenth of the percentile of citation standing between scientists (after window) and for the same scientist.",
        ["Field", "Pairs", "Same scientist, top 5\\%", "Slope, between scientists", "Slope, same scientist"],
        rows,
    )


def first():
    rows = []
    for f, comp in (("first_paper.csv", "Against all other journals"), ("first_paper_neartop.csv", "Against near-top journals")):
        rows.append(comp)
        for _, r in pd.read_csv(SRC / f).iterrows():
            rows.append(
                [
                    FIRST_LABELS[r["sample"]],
                    stack(f"{int(r.treated):,} /", f"{int(r.controls):,}"),
                    f"{100 * r.treated_matched_share:.1f}",
                    f"{n(r.rate_treated_after)} / {n(r.rate_controls_after)}",
                    stack(n(r.att_after), f"({n(r.att_after_lo)} to {n(r.att_after_hi)})"),
                    stack(n(r.att_before), f"({n(r.att_before_lo)} to {n(r.att_before_hi)})"),
                    stack(n(r.did), f"({n(r.did_lo)} to {n(r.did_hi)})"),
                ]
            )
    return table(
        "tab:s_first",
        "First top-journal paper (top 5\\%), index years 2017--2020: matched difference in new lead awards per 1,000 index papers, first top-journal paper minus matched comparison papers, in the after and before windows, and their difference. The all-researchers rows apply the same matching without the restriction to researchers with no earlier top-journal paper.",
        [
            "Sample",
            "Top-journal / comparison papers",
            "Top-journal papers matched (\\%)",
            "Rates before matching, after window (top-journal / comparison)",
            "After window",
            "Before window",
            "After minus before",
        ],
        rows,
    )


def first_checks():
    r = pd.read_csv(SRC / "first_paper_checks.csv")
    r = r[~((r.variant == "primary") & r.window.isin(["months -24 to <0", "months +12 to <36"]))]
    var = {
        "primary": "No earlier top-journal paper",
        "highest earlier percentile from years t-5 to t-3": "Highest earlier percentile of citation standing from years $t-5$ to $t-3$",
        "first OpenAlex paper 2009 or later": "First OpenAlex paper 2009 or later",
    }
    rows = [
        [
            x.contrast.replace("vs ", ""),
            var.get(x.variant, x.variant),
            months(x.window),
            f"{int(x.treated):,} / {int(x.controls):,}",
            ci(x.estimate, x.lo, x.hi),
        ]
        for _, x in r.iterrows()
    ]
    return table(
        "tab:s_first_checks",
        "First top-journal paper: matched difference in new lead awards per 1,000 index papers in other windows and design variants (the before and after windows of the primary comparison are in Table~S11).",
        ["Comparator", "Variant", "Window (months)", "Top-journal / comparison papers", "Difference (95\\% CI)"],
        rows,
        align="lXlrr",
    )


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join([linkage(), between(), timing(), curves(), pairs(), profile(), fields(), first(), first_checks()]))
    print("wrote", OUT)
