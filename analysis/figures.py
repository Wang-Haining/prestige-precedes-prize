"""Figures 1-3 from the aggregate results. python -m analysis.figures [RESULTS_DIR] [FIG_DIR]"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results"
FIG = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "figures"
MM = 1 / 25.4
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
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
BAND, REF = "#EDEDED", "#7A7A7A"
D5 = 7.26  # smallest difference of interest, top 5% against all other journals


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def letter(ax, t, x=-0.14):
    ax.text(x, 1.04, t, transform=ax.transAxes, weight="bold", fontsize=10, va="bottom")


def band(ax, d, horizontal=True):
    (ax.axhspan if horizontal else ax.axvspan)(-d, d, color=BAND, lw=0, zorder=0)
    (ax.axhline if horizontal else ax.axvline)(0, color=REF, lw=0.6, ls="--", zorder=1)


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
    fig, ax = plt.subplots(figsize=(180 * MM, 64 * MM))
    band(ax, D5)
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
    ax.text(0.90, -48.0, "near-top\njournals", fontsize=7.5, ha="center", va="bottom", color=REF)
    ax.plot([0.975], [0], marker="D", ms=3.8, color="#222222", lw=0)
    ax.text(0.975, 2.0, "top 5% of\njournals\n(reference)", fontsize=7.5, ha="center", va="bottom")
    hb = h[(h.comparison == "between") & (h.window == "y_post")].iloc[0]
    hw = h[h.comparison == "within"].iloc[0]
    ax.text(
        0.35,
        -16.5,
        f"Between scientists\nopen: before window, filled: after window\nslope {hb.slope_linear:.2f}",
        color=BETWEEN,
        fontsize=8,
        va="top",
    )
    ax.text(0.20, 9.0, f"Same scientist, after window: slope {hw.slope_linear:.2f}", color=SAME, fontsize=8, va="bottom")
    ax.text(0.06, D5 - 0.4, "smallest difference of interest (±7.26)", fontsize=7.5, color=REF, va="top")
    ax.set_ylim(-51, 13)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Percentile of citation standing in the topic-year set")
    ax.set_ylabel("Awards per 1,000 index papers,\ndifference from the top 5% of journals")
    save(fig, "fig1_curves")


def fig2():
    p = pd.read_csv(SRC / "pairs.csv")
    r = p[(p.field == "all") & (p.definition == "top 5%") & (p.set == "pairs >=24 m")].iloc[0]
    rt = order_rates(r)
    fig, (a, b) = plt.subplots(1, 2, figsize=(180 * MM, 64 * MM), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.5})
    # (a) slope chart: earlier paper -> later paper, by order
    first, second = rt["first"], rt["second"]
    a.plot([0, 1], [first[0], first[1]], color=SAME, lw=1.0)
    a.plot([0, 1], [second[1], second[0]], color=SAME, lw=1.0, ls="--")
    a.plot([0], [first[0]], "D", ms=5, color=SAME)
    a.plot([1], [first[1]], "o", ms=5, color=SAME)
    a.plot([0], [second[1]], "o", ms=5, color=SAME)
    a.plot([1], [second[0]], "D", ms=5, color=SAME)
    a.text(1.08, first[1] - 0.9, f"Top journal first\n(n = {int(r.n_first):,} pairs)", fontsize=7.5, va="center")
    a.text(1.08, second[0] + 0.9, f"Top journal second\n(n = {int(r.n_second):,} pairs)", fontsize=7.5, va="center")
    a.set_xticks([0, 1])
    a.set_xticklabels(["Earlier paper\nof the pair", "Later paper\nof the pair"])
    a.set_xlim(-0.3, 1.9)
    a.set_ylim(55, 72)
    a.set_yticks(range(56, 73, 4))
    a.set_ylabel("Awards per 1,000 index papers,\nafter window")
    a.plot([], [], "D", color=SAME, label="Top-journal paper")
    a.plot([], [], "o", color=SAME, label="Paper in another journal")
    a.legend(frameon=False, fontsize=7.5, loc="lower left", borderaxespad=0.8)
    letter(a, "a", x=-0.3)
    # (b) differences by order and their average
    band(b, D5)
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
    b.text(2.12, e, f"{e:.1f}\n({lo:.1f} to {hi:.1f})".replace("-", "\u2212"), fontsize=7.5, color=SAME, va="center")
    b.set_xticks(range(3))
    b.set_xticklabels([x[0] for x in rows])
    b.set_xlim(-0.6, 2.9)
    b.set_ylabel("Awards per 1,000 index papers,\ntop-journal paper minus paper in another journal")
    b.text(2.85, D5 + 0.6, "±7.26", fontsize=7.5, color=REF, ha="right", va="bottom")
    letter(b, "b", x=-0.3)
    save(fig, "fig2_pairs")


def forest(ax, rows, dl, title):
    band(ax, dl, horizontal=False)
    for i, row in enumerate(rows):
        lab, col, aft, bef = row[:4]
        y = len(rows) - 1 - i
        for vals, dy, face in ((bef, 0.18, "white"), (aft, -0.02, col)):
            e, lo, hi = vals
            ax.errorbar([e], [y + dy], xerr=[[e - lo], [hi - e]], fmt="o", ms=3.8, color=col, mfc=face, elinewidth=0.8, capsize=0)
        if len(row) > 4:
            e, lo, hi = row[4]
            ax.errorbar([e], [y - 0.22], xerr=[[e - lo], [hi - e]], fmt="s", ms=3.4, color=col, elinewidth=0.8, capsize=0)
    ax.axhline(1.5, color="#BBBBBB", lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title(title, fontsize=9)


def fig3():
    s1 = pd.read_csv(SRC / "between.csv")
    p = pd.read_csv(SRC / "pairs.csv")
    p = p[p.field.eq("all") & p.definition.eq("top 5%")].set_index("set")
    d = pd.read_csv(SRC / "first_paper.csv").set_index("sample")
    sn = pd.read_csv(SRC / "between_neartop.csv")
    pn = pd.read_csv(SRC / "pairs_neartop.csv")
    pn = pn[pn.definition.eq("near-top 0.85-0.95")].set_index("set")
    dn = pd.read_csv(SRC / "first_paper_neartop.csv").set_index("sample")

    def rows_for(bt, d_, q, pl):
        r, m = d_.loc["ordinary researchers (primary)"], d_.loc["all researchers, same matching"]
        return [
            (
                "Between scientists\n2014\u20132021",
                BETWEEN,
                (bt.loc["y_post", "rd"], bt.loc["y_post", "rd_lo"], bt.loc["y_post", "rd_hi"]),
                (bt.loc["y_pre", "rd"], bt.loc["y_pre", "rd_lo"], bt.loc["y_pre", "rd_hi"]),
            ),
            (
                "Between scientists,\nmatched, 2017\u20132020",
                BETWEEN,
                (m.att_after, m.att_after_lo, m.att_after_hi),
                (m.att_before, m.att_before_lo, m.att_before_hi),
            ),
            (
                "First top-journal paper,\nmatched, 2017\u20132020",
                FIRST,
                (r.att_after, r.att_after_lo, r.att_after_hi),
                (r.att_before, r.att_before_lo, r.att_before_hi),
                (r["did"], r.did_lo, r.did_hi),
            ),
            ("Same scientist\n2014\u20132021", SAME, (q.rd_bal, q.rd_bal_lo, q.rd_bal_hi), (pl.rd_bal, pl.rd_bal_lo, pl.rd_bal_hi)),
        ]

    bt = s1[(s1.part == "between") & (s1.label == "top 5%")].set_index("window")
    btn = sn[sn.part == "between"].iloc[:2].set_index("window")
    deln = float(sn[sn.part == "delta"].iloc[0].delta_per_1000)
    fig, (a, b) = plt.subplots(1, 2, figsize=(180 * MM, 80 * MM), sharey=True, gridspec_kw={"wspace": 0.08})
    ra = rows_for(bt, d, p.loc["pairs >=24 m"], p.loc["before window, pairs >=60 m"])
    forest(a, ra, D5, "Top 5% minus\nall other journals")
    a.set_yticklabels([x[0] for x in ra][::-1], fontsize=8)
    rb = rows_for(btn, dn, pn.loc["pairs >=24 m"], pn.loc["before window, pairs >=60 m"])
    forest(b, rb, deln, "Top 5% minus near-top journals\n(percentile 0.85 to <0.95)")
    a.plot([], [], "o", color="#444444", mfc="white", label="Before window")
    a.plot([], [], "o", color="#444444", label="After window")
    a.plot([], [], "s", color="#444444", label="After minus before")
    a.legend(frameon=False, fontsize=7.5, loc="upper left")
    for ax in (a, b):
        ax.set_xlabel("Awards per 1,000 index papers")
        ax.set_xlim(-17, 36)
    fig.text(0.0, 0.97, "a", weight="bold", fontsize=10)
    fig.text(0.53, 0.97, "b", weight="bold", fontsize=10)
    save(fig, "fig3_compare")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
