"""Precision of the award linkage in the audited samples (Table S1, author and language-model columns).
-> results/linkage_audit.csv"""

import pandas as pd

from analysis.common import DATA, save

ROWS = {
    "outcome": "Accepted, award in outcome windows",
    "history": "Accepted, award before month -24",
    "missed": "Found only by broad linkage",
}

if __name__ == "__main__":
    x = pd.read_csv(DATA / "link_audit.csv")
    rows = []
    for (sample, quartile), g in x.groupby(["sample", "quartile"], sort=False):
        h = g[g.author_label.notna()]
        rows.append(
            {
                "links": ROWS[sample],
                "quartile": quartile,
                "author_same_person": int(h.author_label.eq("same_person").sum()),
                "author_n": len(h),
                "model_same_person_share": float(g.model_label.eq("same_person").mean()),
                "model_n": len(g),
            }
        )
    save(rows, "linkage_audit.csv")
