"""Lines page with an 11 F + 7 D team: the 7th defenceman shows on the depth chart's pair 4 and
the lineup isn't flagged. Uses the first slate team whose chart has 7 D.

    python -m research.apptest_7d
"""
import os
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import live

s = live.load_state()
hit = None
for day in sorted(s["games"].date.unique()):
    for t, d in live.deployments(s, day, log=False).items():
        pos = dict(zip(s["rosters"][t].player, s["rosters"][t].pos))
        if sum(pos.get(p) == "D" for p in d["skaters"]) == 7:
            hit = day, t
            break
    if hit:
        break
assert hit, "no 11 F + 7 D team on the slate"
day, team = hit
g = s["games"][(s["games"].date == day) & ((s["games"].home == team) | (s["games"].away == team))].iloc[0]

os.environ["APP_PAGE"] = "lines"
at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=300)
at.session_state["day"] = day
at.session_state[f"game_{day}"] = int(g.game_id)
at.session_state[f"lt_{g.game_id}"] = team
at.run()
assert not at.exception, [e.value for e in at.exception]
key = f"{day}{team}D4"
sel = {k: at.selectbox(key=k).value for k in (key + "LD", key + "RD")}
caps = [c.value for c in at.caption if "dressed" in c.value]
html = " ".join(str(h.proto) for h in at.get("html"))
print(day, team, "pair 4:", sel, "|", caps)
assert any(v is not None for v in sel.values()), "7th D missing from pair 4"
assert caps and "usual lineup" not in caps[0], caps
assert "7th" in html, "7th D row missing from the projections chart"
print("ok")
