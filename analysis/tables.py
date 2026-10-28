#!/usr/bin/env python3
"""Supporting tables (S3 onward) from the aggregate results. python -m analysis.tables [RESULTS_DIR] [OUT_TEX]"""

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
    chg = s[s.part.eq("change")].set_index("label")
    rows = ["Top journals against all other journals"]
    for k, lab in DEFS:
        a = b[(b.label == k) & (b.window == "y_post")].iloc[0]
        p = b[(b.label == k) & (b.window == "y_pre")].iloc[0]
        rows.append(
            [
                lab,
                f"{int(a.papers_top):,}",
                f"{n(a.rate_top)} / {n(a.rate_non)}",
                stack(n(a.rd), f"({n(a.rd_lo)} to {n(a.rd_hi)})"),
                stack(n(a.rr, 2), f"({n(a.rr_lo, 2)} to {n(a.rr_hi, 2)})"),
                stack(n(p.rd), f"({n(p.rd_lo)} to {n(p.rd_hi)})"),
                stack(n(chg.loc[k].rr, 2), f"({n(chg.loc[k].rr_lo, 2)} to {n(chg.loc[k].rr_hi, 2)})"),
            ]
        )
    sn = pd.read_csv(SRC / "between_neartop.csv")
    snc = sn[sn.part == "change"].reset_index(drop=True)
    btw = sn[sn.part == "between"].reset_index(drop=True)
    rows.append("Top 5\\% against near-top journals")
    for i, lab in enumerate(["Percentile 0.85 to $<$0.95", "Percentile 0.90 to $<$0.95"]):
        a, p = btw.iloc[2 * i + 1], btw.iloc[2 * i]
        rows.append(
            [
                lab,
                f"{int(a.papers_top):,} / {int(a.papers - a.papers_top):,}",
                f"{n(a.rate_top)} / {n(a.rate_non)}",
                stack(n(a.rd), f"({n(a.rd_lo)} to {n(a.rd_hi)})"),
                stack(n(a.rr, 2), f"({n(a.rr_lo, 2)} to {n(a.rr_hi, 2)})"),
                stack(n(p.rd), f"({n(p.rd_lo)} to {n(p.rd_hi)})"),
                stack(n(snc.iloc[i].rr, 2), f"({n(snc.iloc[i].rr_lo, 2)} to {n(snc.iloc[i].rr_hi, 2)})"),
            ]
        )
    return table(
        "tab:s_between",
        "Between scientists: new lead awards per 1,000 index papers after top-journal papers and after comparison papers, directly standardized over topic-year sets, by definition of a top journal and against near-top journals. In the near-top rows, both rates are standardized over the topic-year sets that contain both kinds of journal. The ratio of risk ratios is the after-window risk ratio over the before-window risk ratio.",
        [
            "Comparison",
            "Top-journal papers (/ near-top papers)",
            "Rates, after (top / comparison)",
            "After: difference",
            "After: risk ratio",
            "Before: difference",
            "Ratio of risk ratios (after / before)",
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
        rows2.append([nm, ci(r.slope_linear, r.slope_linear_lo, r.slope_linear_hi, 2)])
    return table(
        "tab:s_curves",
        "New lead awards per 1,000 index papers by band of the percentile of citation standing, difference from the top band (0.95--1.00) in a linear probability model. Between scientists: topic-year set fixed effects, all index papers. Same scientist: scientist fixed effects and index year, index papers at least 24 months apart.",
        ["Percentile band", "Between, before", "Between, after", "Same scientist, after"],
        rows,
    ) + table(
        "tab:s_hinge",
        "Straight-line fits of new lead awards on the percentile of citation standing, with the fixed effects of Table~S6. Slopes per 1,000 index papers for each tenth of the percentile range. Intervals are from 200 bootstrap replicates.",
        ["Comparison", "Straight-line slope (95\\% CI)"],
        rows2,
        align="lr",
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


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def jp_detection():
    g = pd.read_csv(SRC / "japan_detection.csv")
    rows = []
    for _, x in g.iterrows():
        if x.start_year == g.start_year.min():
            rows.append("Start years 2012--2021: before and after windows of index papers from 2014 to 2018")
        if x.start_year == 2022:
            rows.append("Start years 2022--2024: after windows of index papers from 2019 to 2021 only")
        rows.append(
            [
                str(int(x.start_year)),
                f"{int(x.grants):,}",
                pct(x.share_latin_name),
                pct(x.share_reachable),
                pct(x.share_reachable_early_career),
                pct(x.share_reachable_other),
                pct(x.share_april1),
            ]
        )
    return table(
        "tab:s_jp_detection",
        "KAKEN research grants by start year and the share the linkage can reach. A grant is reachable when its lead investigator's name is recorded in Latin script, or when the lead investigator's researcher number holds a research grant with a Latin-script name. Early-career categories: Young Scientists, Early-Career Scientists, and Research Activity Start-up.",
        [
            "Start year",
            "Research grants",
            "Lead investigator name in Latin script (\\%)",
            "Reachable (\\%)",
            "Reachable, early-career categories (\\%)",
            "Reachable, other categories (\\%)",
            "Started on 1 April (\\%)",
        ],
        rows,
        align="lrrrrrr",
    )


WIN = {"before": "Before window", "after": "After window"}
PAPER = {"top": "top-journal papers", "other": "other papers"}


def jp_audit():
    a = pd.read_csv(SRC / "japan_audit.csv")
    o = a[a.part.eq("outputs") & a.grant_name_script.eq("all")]
    m = a[a.part.eq("model") & a.window.notna()].set_index(["window", "paper"])
    rows = []
    for yrs in ("2014-2018", "2014-2021"):
        rows.append(f"Index papers {yrs.replace('-', '--')}")
        for w in ("before", "after"):
            for pp in ("top", "other"):
                x = o[o.index_years.eq(yrs) & o.window.eq(w) & o.paper.eq(pp)].iloc[0]
                lm = m.loc[(w, pp)] if yrs == "2014-2021" else None
                rows.append(
                    [
                        f"{WIN[w]}, {PAPER[pp]}",
                        f"{int(x.links):,}",
                        f"{int(x.links_with_outputs):,}",
                        pct(x.corroborated),
                        f"{int(x.recall_n):,}",
                        pct(x.recall),
                        "" if lm is None else f"{pct(lm.model_same_person_share)} ({int(lm.model_n)})",
                    ]
                )
    v = a[a.part.eq("model label vs outputs")].set_index("model_label")
    d, sp = v.loc["different_person"], v.loc["same_person"]
    allm = a[a.part.eq("model") & a.window.isna() & a.record_nrids.isna()].iloc[0]
    multi = a[a.part.eq("model") & a.record_nrids.eq(">1")].iloc[0]
    note = (
        " Links with outputs: links whose researcher number lists output papers on any KAKEN grant. On outputs: the linked author is an author of one of those papers;"
        " coauthors also appear on outputs, so this corroborates a link without proving it, and absence does not make a link false."
        " Recovered: grants of lead investigators identifiable from their outputs (exactly one author of the outputs carries the lead investigator's name)"
        " that start in the window and that the linkage attaches to the investigator's index paper."
        f" Language model: Claude Haiku 4.5 labeled 150 outcome-window links per window and group, drawn from index papers of 2014 to 2021 ({pct(allm.model_same_person_share)}\\% same person overall)."
        f" Of its {int(d.model_n)} different-person labels, {int(d.links_with_outputs)} concerned links with outputs, and in {int(d.author_on_outputs)} of these the author was on the outputs"
        f" ({int(sp.author_on_outputs)} of {int(sp.links_with_outputs)} for same-person labels)."
        f" Links of index papers linked to more than one researcher number were labeled the same person in {pct(multi.model_same_person_share)}\\% of {int(multi.model_n)}."
    )
    return table(
        "tab:s_jp_audit",
        "Audit of the Japanese linkage, by window and paper group." + note,
        [
            "Links",
            "Outcome-window links",
            "Links with outputs",
            "On outputs (\\%)",
            "Grants of identifiable lead investigators",
            "Recovered (\\%)",
            "Language model, same person (\\%, n)",
        ],
        rows,
    )


def jp_between():
    b = pd.read_csv(SRC / "japan_between.csv")
    p = pd.read_csv(SRC / "japan_pairs.csv").set_index("sample")
    lab = {"y_pre": "Before window", "y_post": "After window", "after minus before": "After minus before"}
    rows = []
    for rec, head in (
        ("all", "Between scientists"),
        ("one researcher number", "Between scientists, without index papers linked to more than one researcher number"),
    ):
        for yrs in ("2014-2018", "2014-2021"):
            g = b[b.records.eq(rec) & b["sample"].eq(yrs)].set_index("window")
            x = g.iloc[0]
            rows.append(
                f"{head}, index papers {yrs.replace('-', '--')}: {int(x.papers):,} papers by {int(x.scientists):,} scientists,"
                f" {int(x.papers_top):,} in top journals; {int(x.papers_in_sets_with_both):,} in topic-year sets with both kinds of journal"
            )
            for w in ("y_pre", "y_post", "after minus before"):
                r = g.loc[w]
                rows.append(
                    [
                        lab[w],
                        "" if pd.isna(r.rate_top) else f"{n(r.rate_top)} / {n(r.rate_non)}",
                        ci(r.rd, r.rd_lo, r.rd_hi),
                        ci(r.rr, r.rr_lo, r.rr_hi, 2),
                    ]
                )
    rows.append("Same scientist, after window: one pair per scientist, papers at least 24 months apart, average of the two orders")
    for yrs in ("2014-2018", "2014-2021"):
        r = p.loc[yrs]
        rows.append(
            [
                f"Index papers {yrs.replace('-', '--')}, {int(r.pairs):,} pairs ({int(r.n_first):,} / {int(r.n_second):,}); by order {n(r.rd_first)} / {n(r.rd_second)}",
                f"{n(r.rate_top)} / {n(r.rate_non)}",
                ci(r.rd_bal, r.rd_bal_lo, r.rd_bal_hi),
                ci(r.rr_bal, r.rr_bal_lo, r.rr_bal_hi, 2),
            ]
        )
    return table(
        "tab:s_jp_between",
        "Japan: new lead awards per 1,000 index papers after top-journal papers (top 5\\%) and after other papers by Japanese-affiliated last authors. Between scientists, rates are directly standardized over the topic-year sets that contain both kinds of journal, and for the change from the before to the after window the risk-ratio column gives the ratio of the two risk ratios. Same scientist: pairs are listed as top journal first / top journal second, followed by the difference within each order.",
        ["Comparison", "Rates (top-journal / other)", "Difference (95\\% CI)", "Risk ratio (95\\% CI)"],
        rows,
    )


UNIT = "per 1,000 first top-journal papers or comparison papers"
BOOT = "95\\% confidence intervals from 1,000 two-way bootstrap replicates (scientist and journal weights), with matching weights recomputed in each replicate"
CMP = {"all other journals": "All other journals", "near-top journals": "Near-top journals (percentile 0.85 to $<$0.95)"}
EWIN = {
    "months 12..35": "12 to 35",
    "months 36..59": "36 to 59",
    "months 60..83": "60 to 83",
    "months -24..-1": "$-$24 to $-$1",
    "months 0..23": "0 to 23",
}


def bounds():
    """Step-1 comparisons: detection shares, detection-corrected measures and relative upper limits."""
    det = pd.read_csv(SRC / "bounds_detection.csv")
    cor = pd.read_csv(SRC / "bounds_corrected.csv")
    rel = pd.read_csv(SRC / "bounds_relative.csv")
    rows = ["Detection: windows with an acknowledged own award, and the share in which the linkage records a new lead award (\\%)"]
    for (lk, g), x in det.groupby(["linkage", "group"], sort=False):
        b, a = x[x.window == "before window"].iloc[0], x[x.window == "after window"].iloc[0]
        lab = (g[:1].upper() + g[1:]).replace("%", "\\%").replace("2017-2020", "2017--2020") + (" (broad linkage)" if lk == "broad" else "")
        rows.append([lab, f"{int(b.windows):,} / {int(a.windows):,}", f"{100 * b.share:.1f}", f"{100 * a.share:.1f}"])
    keep = [
        "difference, before window",
        "difference, after window",
        "change, after minus before",
        "risk ratio, after window",
        "ratio of risk ratios, after / before",
    ]
    for comp in cor.comparison.unique():
        rows.append(f"{comp[:1].upper() + comp[1:]}: uncorrected and detection-corrected")
        for mz in keep:
            x = cor[(cor.comparison == comp) & (cor.measure == mz)]
            if x.empty:
                continue
            d_ = 2 if "ratio" in mz else 1
            u, c = x[x.correction == "none"].iloc[0], x[x.correction == "detection"].iloc[0]
            rows.append([mz[:1].upper() + mz[1:], ci(u.estimate, u.lo, u.hi, d_), ci(c.estimate, c.lo, c.hi, d_), ""])
    rows.append("Relative measures: estimate and upper 95\\% limit, uncorrected / detection-corrected")
    for (comp, mz), x in rel.groupby(["comparison", "measure"], sort=False):
        u, c = x[x.correction == "none"].iloc[0], x[x.correction == "detection"].iloc[0]
        rows.append(
            [
                f"{comp[:1].upper() + comp[1:]}: {mz}",
                f"{n(u.estimate, 2)} / {n(c.estimate, 2)}",
                f"{n(u.upper_limit, 2)} / {n(c.upper_limit, 2)}",
                "",
            ]
        )
    return table(
        "tab:s_bounds",
        "Detection-corrected and relative bounds for the three comparisons (top 5\\% against all other journals). Each group-window rate is divided by the share of paper windows with an acknowledged own award in which the linkage records a new lead award; in each bootstrap replicate the shares are drawn from their binomial distributions. Differences are new lead awards per 1,000 index papers; intervals as in the main comparisons.",
        ["Measure", "Windows (before / after), uncorrected, or estimate", "Before (\\%), corrected, or upper limit", "After (\\%)"],
        rows,
        align="lrrr",
    )


def bounds_other():
    """Step-1 first-paper comparison with the broad linkage, and the named journals."""
    b = pd.read_csv(SRC / "bounds_broad.csv").iloc[0]
    nm = pd.read_csv(SRC / "bounds_named.csv")
    rows = [f"First top-journal paper, broad linkage: {int(b.treated):,} first top-journal papers, {int(b.controls):,} comparison papers"]
    rows.append(
        [
            "Before window",
            f"{n(b.rate_treated_before)} / {n(b.rate_matched_before)}",
            ci(b.att_before, b.att_before_lo, b.att_before_hi),
            "",
        ]
    )
    rows.append(
        ["After window", f"{n(b.rate_treated_after)} / {n(b.rate_matched_after)}", ci(b.att_after, b.att_after_lo, b.att_after_hi), ""]
    )
    rows.append(["Change, after minus before", "", ci(b.did, b.did_lo, b.did_hi), ""])
    lab = {"y_pre": "Before window", "y_post": "After window", "after minus before": "Change (ratio of risk ratios)"}
    for comp in nm.comparison.unique():
        x = nm[nm.comparison == comp]
        r0 = x.iloc[0]
        if comp == "between scientists":
            head = f"Named journals between scientists: {int(r0.papers_top):,} papers in named journals, {int(r0.papers_other):,} others"
        elif comp == "same scientist":
            head = f"Named journals, same scientist: {int(r0.pairs):,} pairs"
        else:
            head = f"First paper in a named journal: {int(r0.papers_top):,} first papers, matched share {100 * r0.treated_matched_share:.1f}\\%"
        rows.append(head)
        for _, r in x.iterrows():
            rates = "" if pd.isna(r.rate_top) else f"{n(r.rate_top)} / {n(r.rate_other)}"
            rows.append([lab[r.window], rates, ci(r.rd, r.rd_lo, r.rd_hi), ci(r.rr, r.rr_lo, r.rr_hi, 2)])
    return table(
        "tab:s_bounds_other",
        "The first top-journal paper with the broad award linkage, and papers in \\textit{Nature}, \\textit{Science}, \\textit{Cell}, \\textit{New England Journal of Medicine}, \\textit{The Lancet}, \\textit{JAMA} and \\textit{PNAS} (named journals) against all other journals. New lead awards per 1,000 index papers; 95\\% confidence intervals from 1,000 two-way bootstrap replicates.",
        ["Comparison", "Rates (top / other)", "Difference (95\\% CI)", "Risk ratio (95\\% CI)"],
        rows,
    )


def ev(name):
    return pd.read_csv(SRC / f"event_{name}.csv")


def event_cohort():
    el = ev("eligibility").set_index("role")
    la, fl = el.loc["last author"], el.loc["first and last authors"]
    sz = ev("sizes")
    sz = sz[sz.outcome.eq("award")]
    rows = []
    for (cc, cmp_, cells), g in sz.groupby(["cc", "comparator", "cells"], sort=False):
        rows.append(f"{'United States' if cc == 'US' else 'Japan'}, {CMP[cmp_].lower()}, {cells} cells")
        for (smp, v), x in g.groupby(["sample", "variant"], sort=False):
            t, c = x[x.arm == "treated"].iloc[0], x[x.arm == "control"].iloc[0]
            rows.append(
                [
                    f"{smp.replace('-', '--')}, cells {v}",
                    f"{int(t.treated_matched):,} of {int(t.treated_in_sample):,}",
                    f"{int(c.trials):,} ({int(c.scientists):,})",
                    f"{int(t.followed_to_36):,} / {int(t.followed_to_60):,} / {int(t.followed_to_84):,}",
                    f"{int(c.censored_top_paper):,}",
                    f"{int(t.exits_0_84):,} / {int(c.exits_0_84):,}",
                ]
            )
    return table(
        "tab:s_event_cohort",
        f"Event study: cohort sizes, follow-up and censoring. Of {int(la.eligible_treated):,} first top-journal papers by eligible US last authors, 2014--2021, {int(la.matched_treated):,} ({100 * la.matched_treated / la.eligible_treated:.1f}\\%) had a comparison paper in their month cell A; across first and last authors, the papers without one were more often by researchers with an award started before month $-24$ ({100 * fl.funded_before_m24_unmatched:.1f}\\% against {100 * fl.funded_before_m24_matched:.1f}\\%). Below, treated counts are given within that matched cohort. A trial is a last author's month with a ranked paper and no earlier top-journal paper (first or last author, from 2009). Matched treated trials are those in cells with a comparison trial; comparison trials are capped at 20 per treated trial in each cell. Follow-up columns give treated trials observed beyond 36 and 60 months and through 84 months; comparison trials are censored when they publish a first top-journal paper; exits are trials whose last article or review falls before censoring and is followed by more than 24 months without one.",
        [
            "Sample",
            "Treated trials matched",
            "Comparison trials (scientists)",
            "Treated followed to 36 / 60 / 84 months",
            "Comparison censored by a top-journal paper",
            "Exits, treated / comparison",
        ],
        rows,
        align="lrrrrr",
    )


def event_cif():
    h = ev("horizons")
    h = h[h.outcome.eq("award") & h.measure.eq("cif")]
    b = ev("before")
    b = b[b.window.eq("months -24..-1")].set_index(["sample", "variant", "cells", "comparator"])
    rows = []
    for (_cc, cmp_, cells, smp, v), g in h.groupby(["cc", "comparator", "cells", "sample", "variant"], sort=False):
        rows.append(f"{smp.replace('-', '--')}; {CMP[cmp_].lower()}; {cells} cells {v}")
        if (smp, v, cells, cmp_) in b.index:
            r = b.loc[(smp, v, cells, cmp_)]
            rows.append(
                [
                    "Before window, months $-24$ to $-1$",
                    f"{n(1000 * r.treated)} / {n(1000 * r.control)}",
                    "matched" if abs(r.diff_hi) < 1e-9 else ci(1000 * r["diff"], 1000 * r.diff_lo, 1000 * r.diff_hi),
                    "" if abs(r.diff_hi) < 1e-9 else f"{100 * r.rel:.0f}",
                ]
            )
        for _, r in g.iterrows():
            rows.append(
                [
                    f"{int(r.horizon)} months",
                    f"{n(1000 * r.treated)} / {n(1000 * r.control)}",
                    ci(1000 * r["diff"], 1000 * r.diff_lo, 1000 * r.diff_hi),
                    ci(100 * r.rel, 100 * r.rel_lo, 100 * r.rel_hi, 0),
                ]
            )
    return table(
        "tab:s_event_cif",
        f"Event study: cumulative incidence of a new lead award (Aalen--Johansen, exit competing) {UNIT}, first top-journal papers against matched comparison papers. Cells A match on the prior record to month $-24$ and on awards in months $-60$ to $-24$; cells B also on any new award in months $-24$ to $-1$. Comparison trials are weighted by the inverse probability of not yet being censored by their own first top-journal paper. The before-window row gives the share with a new lead award starting in months $-24$ to $-1$. Relative differences are in percent of the comparison incidence. {BOOT}.",
        ["Horizon", "Incidence (first top-journal / comparison)", "Difference (95\\% CI)", "Relative difference, \\% (95\\% CI)"],
        rows,
    )


def event_change():
    c = ev("change")
    keep = [
        "difference before",
        "difference after",
        "change (after minus before)",
        "ratio of risk ratios (after / before)",
        "change / control after-window rate",
    ]
    rows = []
    for (_cc, cmp_, cells, corr), g in c.groupby(["cc", "comparator", "cells", "correction"], sort=False):
        smp = g["sample"].iloc[0]
        rows.append(
            f"{smp.replace('-', '--')}; {CMP[cmp_].lower()}; {cells} cells; {'uncorrected' if corr == 'none' else 'detection-corrected'}"
        )
        for wn, x in g.groupby("after_window", sort=False):
            v = x.set_index("measure")
            cell = [f"{EWIN[wn]} ({int(x.treated_trials.iloc[0]):,} / {int(x.control_trials.iloc[0]):,})"]
            for mz in keep:
                r = v.loc[mz]
                sc, d_ = (1000, 1) if mz.startswith(("difference", "change (")) else (1, 2)
                cell.append(stack(n(sc * r.estimate, d_), f"({n(sc * r.lo, d_)}, {n(sc * r.hi, d_)})"))
            rows.append(cell)
    return table(
        "tab:s_event_change",
        f"Event study: change from the before window (months $-24$ to $-1$) to each after window, on trials observed through the end of the after window, cells A. Differences and changes are new lead awards {UNIT} in the window (any new award, no exit or censoring); the ratio of risk ratios is the after-window risk ratio over the before-window risk ratio; the last column is the change over the comparison rate in the after window. Detection-corrected rows divide each arm-window rate by its detection share (Table~\\ref{{tab:s_event_detection}}), drawn binomially in each replicate. {BOOT}.",
        [
            "After window (treated / comparison trials)",
            "Difference before",
            "Difference after",
            "Change",
            "Ratio of risk ratios",
            "Change / comparison rate",
        ],
        rows,
        align="lrrrrr",
    )


def event_detection():
    d = ev("detection")
    rows = []
    for (cmp_, cells), g in d.groupby(["comparator", "cells"], sort=False):
        rows.append(f"{g['sample'].iloc[0].replace('-', '--')}; {CMP[cmp_].lower()}; {cells} cells")
        for wn, x in g.groupby("after_window", sort=False):
            v = x.set_index(["arm", "window"])
            rows.append(
                [EWIN[wn]]
                + [
                    f"{int(v.loc[(a, w)].trial_windows_with_reference):,}: {100 * v.loc[(a, w)].share:.1f}"
                    for a in ("treated", "control")
                    for w in ("before", "after")
                ]
            )
    return table(
        "tab:s_event_detection",
        "Event study: detection of the outcome. Among trial windows with an award acknowledged in the author's own papers and led by the author (US last authors whose index records were in the top or bottom quartile of journals), the number of windows and the share (\\%) in which the linkage records a new lead award, by arm and window, on the trials of each after-window sample.",
        ["After window", "Treated, before", "Treated, after", "Comparison, before", "Comparison, after"],
        rows,
        align="lrrrr",
    )


def event_secondary():
    h = ev("horizons")
    h = h[h.variant.eq("A") & h.cells.eq("month") & h.comparator.eq("all other journals") & h.horizon.isin([36, 84])]
    spec = [
        ("award", "active_funded", "Active and funded", 1000, 1),
        ("award", "exit", "Exit", 1000, 1),
        ("award_noexit", "cif", "New award, exit not competing", 1000, 1),
        ("award", "dollars", "Mean amount of new awards started (US\\$ or yen, thousands)", 1e-3, 1),
        ("dollars_ge_500000", "cif", "Cumulative amount $\\geq$ US\\$500,000", 1000, 1),
        ("dollars_ge_1000000", "cif", "Cumulative amount $\\geq$ US\\$1,000,000", 1000, 1),
        ("dollars_ge_5000000", "cif", "Cumulative amount $\\geq$ 5 million yen", 1000, 1),
        ("dollars_ge_20000000", "cif", "Cumulative amount $\\geq$ 20 million yen", 1000, 1),
        ("renewal", "cif", "NIH competing renewal, holders of a renewable project", 1000, 1),
    ]
    rows = []
    for smp, g in h.groupby("sample", sort=False):
        rows.append(smp.replace("-", "--"))
        for o, mz, lab, sc, d_ in spec:
            for hz in (36, 84):
                x = g[(g.outcome == o) & (g.measure == mz) & (g.horizon == hz)]
                if x.empty:
                    continue
                r = x.iloc[0]
                rows.append(
                    [
                        f"{lab}, {hz} months",
                        f"{n(sc * r.treated, d_)} / {n(sc * r.control, d_)}",
                        ci(sc * r["diff"], sc * r.diff_lo, sc * r.diff_hi, d_),
                        ci(100 * r.rel, 100 * r.rel_lo, 100 * r.rel_hi, 0),
                    ]
                )
    return table(
        "tab:s_event_secondary",
        f"Event study: secondary outcomes, cells A (month), against all other journals. Rates {UNIT}; amounts are means per trial of the first-budget-year amounts of new NIH awards and the award amounts of NSF awards (Japan: KAKEN grant amounts), counted until exit. Renewal: a type-2 segment of a linked NIH project (R01, R15, R24, R35, R37, U01) among trials holding one at time zero; its start is inferred from the end of the preceding support year and may be off by more than two months for about a quarter to a third of renewals. NIH amounts (first budget year) and NSF amounts (whole award) are on different scales and are summed as dollars started. None of these outcomes has a before-window comparison. {BOOT}.",
        ["Measure", "First top-journal / comparison", "Difference (95\\% CI)", "Relative, \\% (95\\% CI)"],
        rows,
    )


def event_validation():
    v = ev("validation").set_index(["sample", "measure"])
    rows = []
    for smp in v.index.get_level_values(0).unique():
        x = v.loc[smp]
        r0 = x.iloc[0]
        lab = "First-paper comparison, index papers 2017--2020" if smp.startswith("first_paper") else smp.replace("-", "--")
        rows.append(f"{lab} ({int(r0.treated_trials):,} first top-journal papers, {int(r0.control_trials):,} comparison papers)")
        for mz in ("difference before", "difference after", "change (after minus before)"):
            r = x.loc[mz]
            rows.append([mz[:1].upper() + mz[1:], ci(1000 * r.estimate, 1000 * r.lo, 1000 * r.hi)])
    return table(
        "tab:s_event_validation",
        f"Event study restricted to trials of 2017--2020 and the three-year windows (before: months $-24$ to $-1$; after: months 12 to 35), cells A (month), beside the first-paper comparison of index papers of 2017--2020 matched within index year (Table~S11). New lead awards {UNIT}. {BOOT}.",
        ["Sample and measure", "Estimate (95\\% CI)"],
        rows,
        align="lr",
    )


def split():
    t = pd.read_csv(SRC / "split_type_domain.csv")
    grp = {
        "named seven": "Seven named journals",
        "field top 5% (not named)": "Top 5\\%, outside the named journals",
        "field no. 1 (not named)": "Most cited journal of the set, outside the named journals",
        "no. 1, all": "Most cited journal of the set",
        "NEJM": "\\textit{New England Journal of Medicine}",
    }
    rows = []
    for dom, g in t.groupby("domain", sort=False):
        rows.append("All domains" if dom == "all" else dom)
        for _, r in g.iterrows():
            rows.append(
                [
                    grp[r.group],
                    f"{int(r.papers_top):,}",
                    f"{n(r.rate_top_pre)} / {n(r.rate_non_pre)}",
                    f"{n(r.rate_top_post)} / {n(r.rate_non_post)}",
                    ci(r.rrr, r.rrr_lo, r.rrr_hi, 2),
                ]
            )
    return table(
        "tab:s_split",
        "Between scientists by kind of top journal and by OpenAlex domain of the paper's primary topic: new lead awards per 1,000 index papers before and after publication, directly standardized over topic-year sets, and the ratio of the after-window to the before-window risk ratio. The comparator is papers below the top 5\\% of their set in journals other than the seven named journals, so the rows differ from Table~S4. Groups overlap, and cells with fewer than 200 papers in the top group were not estimated. Intervals are pointwise, from 1,000 two-way bootstrap replicates.",
        [
            "Top group",
            "Papers in the top group",
            "Before (top / comparator)",
            "After (top / comparator)",
            "Ratio of risk ratios (95\\% CI)",
        ],
        rows,
        align="lrrrr",
    )


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "\n".join(
            [
                linkage(),
                between(),
                timing(),
                curves(),
                pairs(),
                profile(),
                fields(),
                first(),
                first_checks(),
                jp_detection(),
                jp_audit(),
                jp_between(),
                bounds(),
                bounds_other(),
                event_cohort(),
                event_cif(),
                event_change(),
                event_detection(),
                event_secondary(),
                event_validation(),
                split(),
            ]
        )
    )
    print("wrote", OUT)
