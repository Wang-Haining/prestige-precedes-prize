"""Figures 1-4 from the aggregate results. python -m analysis.figures [RESULTS_DIR] [FIG_DIR]"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"
FIG = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "figures"
MM = 1 / 25.4
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": "#262626",
        "axes.labelcolor": "#262626",
        "xtick.color": "#262626",
        "ytick.color": "#262626",
    }
)
BETWEEN, SAME, FIRST = "#5B2A5E", "#0F8A6C", "#C0603A"  # between scientists, same scientist, first top-journal paper
REF = "#7A7A7A"
WIDTH = 174  # mm, Springer single-column width
# type scale (pt): panel letter 8 bold, panel title 7 bold, labels and annotations 6.5, ticks 6
LETTER, TITLE, TEXT = 8, 7, 6.5


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def title(ax, text, letter_x=-0.14, pad=6):
    """Conclusion title, left-aligned over the axes, with the panel letter beside its first line."""
    ax.set_title(text, loc="left", fontsize=TITLE, fontweight="bold", pad=pad)
    if letter_x is not None:
        dy = pad + text.count("\n") * TITLE * 1.2  # baseline of the title's first line, in points
        t = matplotlib.transforms.offset_copy(ax.transAxes, fig=ax.figure, y=dy, units="points")
        ax.text(letter_x, 1.0, LETTER_OF[id(ax)], transform=t, weight="bold", fontsize=LETTER, va="baseline")


LETTER_OF = {}


def panel(ax, t):
    LETTER_OF[id(ax)] = t
    return ax


def zero(ax, at=0, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(at, color=REF, lw=0.6, ls="--", zorder=1)


def order_rates(r):
    """Per-order award rates recovered exactly from the order-specific difference and ratio."""
    out = {}
    for o in ("first", "second"):
        non = r[f"rd_{o}"] / (r[f"rr_{o}"] - 1)
        out[o] = (non * r[f"rr_{o}"], non)  # (after top-journal paper, after other paper)
    return out


def fig1():
    c = pd.read_csv(SRC / "curves.csv")
    c = c[c.field.eq("all")]
    h = pd.read_csv(SRC / "hinge.csv")
    h = h[h.field.eq("all")]
    fig, ax = plt.subplots(figsize=(WIDTH * MM, 66 * MM))
    zero(ax)
    specs = [
        ("between", "y_pre", BETWEEN, "white", -0.005),
        ("between", "y_post", BETWEEN, BETWEEN, 0.0),
        ("within", "y_post", SAME, SAME, 0.005),
    ]
    for comp, w, col, face, dx in specs:
        s = c[(c.comparison == comp) & (c.window == w)].sort_values("pct_lo")
        x = (s.pct_lo + s.pct_hi) / 2 + dx
        ax.errorbar(
            x,
            s.rd_vs_top,
            yerr=[s.rd_vs_top - s.lo, s.hi - s.rd_vs_top],
            fmt="o",
            ms=3.2,
            mfc=face,
            mec=col,
            color=col,
            elinewidth=0.6,
            capsize=0,
            lw=0,
        )
        ax.plot(list(x) + [0.975], list(s.rd_vs_top) + [0], color=col, lw=0.6, alpha=0.6)
    ax.plot([0.85, 0.85, 0.95, 0.95], [-48.2, -49.2, -49.2, -48.2], color=REF, lw=0.7, clip_on=False)
    ax.text(0.90, -48.0, "near-top\njournals", fontsize=TEXT, ha="center", va="bottom", color=REF)
    ax.plot([0.975], [0], marker="D", ms=3.8, color="#222222", lw=0)
    ax.text(0.975, 2.0, "top 5% of\njournals\n(reference)", fontsize=TEXT, ha="center", va="bottom")
    hb = h[(h.comparison == "between") & (h.window == "y_post")].iloc[0]
    hw = h[h.comparison == "within"].iloc[0]
    ax.text(
        0.35,
        -16.5,
        f"Between scientists\nopen: before window, filled: after window\nslope {hb.slope_linear:.2f}",
        color=BETWEEN,
        fontsize=TEXT,
        va="top",
    )
    ax.text(0.20, 9.0, f"Same scientist, after window: slope {hw.slope_linear:.2f}", color=SAME, fontsize=TEXT, va="bottom")
    ax.set_ylim(-51, 13)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Percentile of citation standing in the topic-year set")
    ax.set_ylabel("Awards per 1,000 index papers,\ndifference from the top 5% of journals")
    title(
        ax,
        "Between scientists, award rates climb above the 85th percentile of citation standing; for the same scientist they barely change",
        letter_x=None,
    )
    save(fig, "fig1_curves")


def fig2():
    p = pd.read_csv(SRC / "pairs.csv")
    r = p[(p.field == "all") & (p.definition == "top 5%") & (p.set == "pairs >=24 m")].iloc[0]
    rt = order_rates(r)
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH * MM, 66 * MM), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.5})
    panel(a, "a")
    panel(b, "b")
    # (a) slope chart: earlier paper -> later paper, by order
    first, second = rt["first"], rt["second"]
    a.plot([0, 1], [first[0], first[1]], color=SAME, lw=1.0)
    a.plot([0, 1], [second[1], second[0]], color=SAME, lw=1.0, ls="--")
    a.plot([0], [first[0]], "D", ms=5, color=SAME)
    a.plot([1], [first[1]], "o", ms=5, color=SAME)
    a.plot([0], [second[1]], "o", ms=5, color=SAME)
    a.plot([1], [second[0]], "D", ms=5, color=SAME)
    a.text(1.08, first[1] - 0.9, f"Top journal first\n(n = {int(r.n_first):,} pairs)", fontsize=TEXT, va="center")
    a.text(1.08, second[0] + 0.9, f"Top journal second\n(n = {int(r.n_second):,} pairs)", fontsize=TEXT, va="center")
    a.set_xticks([0, 1])
    a.set_xticklabels(["Earlier paper\nof the pair", "Later paper\nof the pair"])
    a.set_xlim(-0.3, 1.9)
    a.set_ylim(55, 72)
    a.set_yticks(range(56, 73, 4))
    a.set_ylabel("Awards per 1,000 index papers,\nafter window")
    a.plot([], [], "D", color=SAME, label="Top-journal paper")
    a.plot([], [], "o", color=SAME, label="Paper in another journal")
    a.legend(frameon=False, fontsize=TEXT, loc="lower left", borderaxespad=0.8)
    title(a, "The earlier paper of a pair is followed by more\nawards, whichever journal it appeared in", letter_x=-0.3)
    # (b) differences by order and their average
    zero(b)
    rows = [
        ("Top journal\nfirst", r.rd_first, r.rd_first_lo, r.rd_first_hi, 3.6),
        ("Top journal\nsecond", r.rd_second, r.rd_second_lo, r.rd_second_hi, 3.6),
        ("Average of\nboth orders", r.rd_bal, r.rd_bal_lo, r.rd_bal_hi, 5.0),
    ]
    for i, (_lab, e, lo, hi, _ms) in enumerate(rows):
        main = i == 2
        b.errorbar(
            [i],
            [e],
            yerr=[[e - lo], [hi - e]],
            fmt="o",
            ms=5.6 if main else 3.6,
            color=SAME,
            alpha=1.0 if main else 0.5,
            elinewidth=1.4 if main else 0.8,
            capsize=0,
        )
    b.axvline(1.5, color="#BBBBBB", lw=0.6)
    e, lo, hi = rows[2][1:4]
    b.text(2.12, e, f"{e:.1f}\n({lo:.1f} to {hi:.1f})".replace("-", "\u2212"), fontsize=TEXT, va="center")
    b.set_xticks(range(3))
    b.set_xticklabels([x[0] for x in rows])
    b.set_xlim(-0.6, 2.9)
    b.set_ylabel("Awards per 1,000 index papers,\ntop-journal paper minus paper in another journal")
    title(
        b,
        "Averaged over both orders, award rates after top-journal\nand other papers differ by "
        + f"{e:.1f}".replace("-", "\u2212")
        + " per 1,000",
        letter_x=-0.3,
    )
    save(fig, "figS1_pairs")


def ev(name, sample, comparator="all other journals"):
    t = pd.read_csv(SRC / f"event_{name}.csv")
    return t[t["sample"].eq(sample) & t.comparator.eq(comparator)]


US, JP = "US 2014-2021, last authors", "JP 2014-2018, last authors"
WINDOWS = ["months 12..35", "months 36..59", "months 60..83"]


def change(sample, comparator="all other journals", correction="none"):
    """Ratio of risk ratios (after / before) by after window, month A cells."""
    t = ev("change", sample, comparator)
    t = t[t.cells.eq("month A") & t.correction.eq(correction) & t.measure.eq("ratio of risk ratios (after / before)")]
    return t.set_index("after_window").loc[WINDOWS, ["estimate", "lo", "hi"]]


def ratio_axis(ax, lo=0.7, hi=1.45, horizontal=False):
    """Log ratio axis with a reference line at 1 (no widening)."""
    zero(ax, 1, horizontal=horizontal)
    ticks = [0.75, 1, 1.25]
    if horizontal:
        ax.set_yscale("log"), ax.set_yticks(ticks), ax.set_yticklabels([f"{x:g}" for x in ticks]), ax.set_ylim(lo, hi)
    else:
        ax.set_xscale("log"), ax.set_xticks(ticks), ax.set_xticklabels([f"{x:g}" for x in ticks]), ax.set_xlim(lo, hi)
    ax.minorticks_off()


def fig3():
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH * MM, 70 * MM), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.42})
    panel(a, "a")
    panel(b, "b")
    # (a) cumulative incidence, US last authors, main matching
    c = ev("curves", US)
    c = c[c.variant.eq("A") & c.cells.eq("month") & c.outcome.eq("award") & c.measure.eq("cif") & c.month.le(84)].sort_values("month")
    for arm, col, lab in (("treated", FIRST, "First top-journal paper"), ("control", REF, "Matched comparison paper")):
        a.fill_between(c.month, 1000 * c[f"{arm}_lo"], 1000 * c[f"{arm}_hi"], color=col, alpha=0.18, lw=0)
        a.plot(c.month, 1000 * c[arm], color=col, lw=1.2, label=lab)
    bf = ev("before", US)
    bf = bf[bf.variant.eq("A") & bf.cells.eq("month") & bf.window.eq("months -24..-1")].iloc[0]
    end = c[c.month.eq(84)].iloc[0]
    a.text(
        2,
        58,
        f"Before the paper (months −24 to −1):\n{1000 * bf.treated:.1f} against {1000 * bf.control:.1f} per 1,000 "
        f"(+{100 * (bf.treated / bf.control - 1):.0f}%)",
        fontsize=TEXT,
        va="top",
    )
    a.text(86, 1000 * end.treated, f"{1000 * end.treated:.1f}", fontsize=TEXT, va="center", color=FIRST)
    a.text(86, 1000 * end.control, f"{1000 * end.control:.1f}", fontsize=TEXT, va="center", color=REF)
    a.set_xlim(0, 96)
    a.set_xticks(range(0, 85, 12))
    a.set_ylim(0, 60)
    a.set_xlabel("Months after publication")
    a.set_ylabel("Cumulative incidence of a new lead award\nper 1,000 papers")
    a.legend(frameon=False, fontsize=TEXT, loc="lower right")
    title(a, "After a first top-journal paper, new awards accrue\nabout a fifth more often, as they did before it", letter_x=-0.22)
    # (b) ratio of risk ratios by after window
    series = [
        ("Linked awards", change(US), "o", 1.0),
        ("Corrected for detection", change(US, correction="detection"), "^", 1.0),
    ]
    off = [-0.1, 0.1]
    ratio_axis(b, 0.62, 1.75, horizontal=True)
    for k, (lab, t, m, al) in enumerate(series):
        x = [i + off[k] for i in range(len(WINDOWS))]
        b.errorbar(
            x,
            t.estimate,
            yerr=[t.estimate - t.lo, t.hi - t.estimate],
            fmt=m,
            ms=3.8,
            color=FIRST,
            alpha=al,
            elinewidth=0.8,
            capsize=0,
            label=lab,
        )
    b.set_yticks([0.75, 1, 1.25, 1.5])
    b.set_yticklabels(["0.75", "1", "1.25", "1.5"])
    b.set_xticks(range(len(WINDOWS)))
    b.set_xticklabels(["Months 12–35", "Months 36–59", "Months 60–83"])
    b.set_xlim(-0.5, len(WINDOWS) - 0.5)
    b.set_ylabel("Risk ratio after the paper / risk ratio before it\n(log scale; 1 = no widening)")
    b.legend(frameon=False, fontsize=TEXT, loc="upper left", handlelength=1.2)
    title(b, "In no window up to seven years did the\nrelative lead grow", letter_x=-0.22)
    save(fig, "fig2_event")


def fig4():
    rel = pd.read_csv(SRC / "bounds_corrected.csv")

    def bc(comp, measure, corr):
        r = rel[rel.comparison.eq(comp) & rel.measure.eq(measure) & rel.correction.eq(corr)].iloc[0]
        return r.estimate, r.lo, r.hi

    s1 = pd.read_csv(SRC / "between.csv")
    jp = pd.read_csv(SRC / "japan_between.csv")
    jp = jp[jp["sample"].eq("2014-2018") & jp.records.eq("all")].set_index("window")

    def first(comparator, corr="none"):
        return tuple(change(US, comparator, corr).loc["months 12..35"])

    rr_ = "ratio of risk ratios, after / before"
    rows_a = [
        ("Between scientists,\nUnited States", BETWEEN, bc("between scientists", rr_, "none"), bc("between scientists", rr_, "detection")),
        ("Between scientists,\nJapan", BETWEEN, tuple(jp.loc["after minus before", ["rr", "rr_lo", "rr_hi"]]), None),
        ("First top-journal paper,\nmonths 12–35", FIRST, first("all other journals"), first("all other journals", "detection")),
        (
            "Same scientist,\nrisk ratio after the paper",
            SAME,
            bc("same scientist", "risk ratio, after window", "none"),
            bc("same scientist", "risk ratio, after window", "detection"),
        ),
    ]
    fig = plt.figure(figsize=(WIDTH * MM, 112 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.55], wspace=0.08, hspace=0.75)
    a = panel(fig.add_subplot(gs[0, :]), "a")
    c = panel(fig.add_subplot(gs[1, 0]), "b")
    key = fig.add_subplot(gs[1, 1])
    n = len(rows_a)
    for ax, rows in ((a, rows_a),):
        ratio_axis(ax, 0.7, 1.45)
        for i, (_lab, col, e, ec) in enumerate(rows):
            y = n - 1 - i
            for vals, dy, m in ((e, 0.1, "o"), (ec, -0.14, "^")):
                if vals is None:
                    continue
                v, lo, hi = vals
                ax.errorbar([v], [y + dy], xerr=[[v - lo], [hi - v]], fmt=m, ms=3.8, color=col, elinewidth=0.8, capsize=0)
        ax.axhline(0.5, color="#BBBBBB", lw=0.6)  # above: different scientists; below: the same scientist
        ax.set_ylim(-0.6, n - 0.4)
    a.set_yticks(range(n))
    a.set_yticklabels([x[0] for x in rows_a][::-1])
    a.set_xlabel(
        "Risk ratio after publication / risk ratio before it; same scientist: risk ratio after the paper\n(log scale; 1 = no change)"
    )
    title(a, "With top journals as the top 5% of each topic-year set, no comparison shows the risk ratio growing", letter_x=-0.3)
    # (c) risk ratios between scientists, United States and Japan
    us = s1[s1.label.eq("top 5%") & s1.part.eq("between")].set_index("window")
    rc = [("United States\n2014–2021", us), ("Japan\n2014–2018", jp)]
    zero(c, 1, horizontal=False)
    for i, (_lab, t) in enumerate(rc):
        y = len(rc) - 1 - i
        for w, dy, face in (("y_pre", 0.12, "white"), ("y_post", -0.12, BETWEEN)):
            e, lo, hi = t.loc[w, "rr"], t.loc[w, "rr_lo"], t.loc[w, "rr_hi"]
            c.errorbar([e], [y + dy], xerr=[[e - lo], [hi - e]], fmt="o", ms=3.8, color=BETWEEN, mfc=face, elinewidth=0.8, capsize=0)
    c.set_xscale("log")
    c.set_xticks([0.8, 1, 1.25, 1.5, 2, 2.5])
    c.set_xticklabels(["0.8", "1", "1.25", "1.5", "2", "2.5"])
    c.minorticks_off()
    c.set_xlim(0.78, 2.6)
    c.set_yticks(range(len(rc)))
    c.set_yticklabels([x[0] for x in rc][::-1])
    c.set_ylim(-0.5, len(rc) - 0.5)
    c.set_xlabel("Risk ratio between scientists, top 5% to all other journals (log scale)")
    title(
        c,
        "Top-journal papers mark about twice the award rate in the US\nand a tenth more in Japan, before as after publication",
        letter_x=-0.62,
    )
    key.axis("off")
    grey = "#444444"
    handles = [
        Line2D([], [], marker="o", ls="", color=grey, ms=4, label="Linked awards (a)"),
        Line2D([], [], marker="^", ls="", color=grey, ms=4, label="Corrected for detection (a)"),
        Line2D([], [], marker="o", ls="", color=grey, mfc="white", ms=4, label="Before window, months −24 to <0 (b)"),
        Line2D([], [], marker="o", ls="", color=grey, ms=4, label="After window, months +12 to <36 (b)"),
        Line2D([], [], color=BETWEEN, lw=1.6, label="Between scientists"),
        Line2D([], [], color=FIRST, lw=1.6, label="First top-journal paper, matched"),
        Line2D([], [], color=SAME, lw=1.6, label="Same scientist"),
    ]
    key.legend(handles=handles, frameon=False, loc="center left", fontsize=TEXT, handlelength=1.6)
    save(fig, "fig3_bounds")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
