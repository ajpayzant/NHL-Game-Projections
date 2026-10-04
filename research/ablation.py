"""Which data groups carry information the structured model misses? For each target, boost
the structured model's errors with ONE group at a time (train 2023-24, test 2025-26), and list
the top single variables by gain when every group is allowed."""
import json, sys
import numpy as np, pandas as pd
import config as C
from research.models import TARGETS, fit_lgb, score

m = pd.read_parquet(C.DATA / "research_matrix.parquet")
groups = {g: c.split(",") for g, c in json.loads((C.DATA / "research_groups.json").read_text()).items()}
m["toi_min"] = m.toi / 60
m = m[(m.toi > 0) & (m.gp_16 > 0) & (m.season >= 2023)].reset_index(drop=True)
tr, te = m[m.season.between(2023, 2024)], m[m.season == 2025]
allf = [c for cs in groups.values() for c in cs]
rows = []
for tname in sys.argv[1:] or TARGETS:
    y, s = TARGETS[tname]
    is_toi = tname == "toi"
    link = (lambda v: v) if is_toi else (lambda v: np.log(np.clip(v, 1e-4, None)))
    base = score(None, te[y].to_numpy(), te[s].to_numpy(), is_toi)
    for g, cols in list(groups.items()) + [("ALL", allf)]:
        p, mdl, _ = fit_lgb(tr[cols].to_numpy(), tr[y].to_numpy(), te[cols].to_numpy(), is_toi,
                            init_tr=link(tr[s].to_numpy()), init_te=link(te[s].to_numpy()))
        sc = score(None, te[y].to_numpy(), p, is_toi)
        rows.append({"target": tname, "group": g, "gain_%": 100 * (base - sc) / base})
        if g == "ALL":
            imp = pd.Series(mdl.feature_importance("gain"), index=cols).sort_values(ascending=False)
            print(tname, "top variables:", ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.head(10).items()), flush=True)
r = pd.DataFrame(rows).pivot(index="group", columns="target", values="gain_%")
print("\n% improvement over the structured model from boosting its errors with one data group (2025-26)")
print(r.round(2).to_string())
