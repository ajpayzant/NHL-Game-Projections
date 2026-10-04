import json
import backtest, model
for lines in ("last", "tonight"):
    f = backtest.feature_table(lines)
    co = model.fit(f[f.season.between(2022, 2025)])
    print(lines, json.dumps({k: {kk: round(vv, 3) for kk, vv in co[k].items()} for k in ("sog", "g", "a", "alpha")}))
    print("  share", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in co["toi"]["share"].items()}))
