"""Headless check of the Player page for goalies: python -m research.apptest_goalie_player
Picks each goalie on the latest slate through session state (AppTest can't drive a
selectbox with format_func) and renders his page."""
import os
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
os.environ["APP_PAGE"] = "player"
at = AppTest.from_file(APP, default_timeout=300)
at.run()
box = next(b for b in at.sidebar.selectbox if b.label == "Player")
goalies = [o for o in box.options if "· G" in o]
print("goalies in list:", len(goalies), goalies[:3])
for j in range(4):
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state[box.key] = f"g{j}"
    at.run()
    print(f"g{j} exceptions: {[e.value for e in at.exception][:1]} | tables: {len(at.dataframe)} "
          f"| metrics: {len(at.metric)} | title: {[t.value for t in at.title]}", flush=True)
