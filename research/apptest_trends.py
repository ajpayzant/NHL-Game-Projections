"""Headless check of the Trends page, every tab and control: python -m research.apptest_trends"""
import os
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
os.environ["APP_PAGE"] = "trends"
at = AppTest.from_file(APP, default_timeout=300)
at.run()
print("exceptions:", [e.value for e in at.exception][:3], "| errors:", [e.value for e in at.error])
print("tables:", len(at.dataframe), [len(d.value) for d in at.dataframe])
for d in at.dataframe:
    print(d.value.head(3).to_string(), "\n")
# the controls: depth players on, hit rates on shots o2.5 over 20 games, the season team window, career milestones
at.toggle(key="tr_depth").set_value(True)
at.segmented_control(key="tr_hstat").set_value("Shots on goal")
at.segmented_control(key="tr_hwin").set_value("Last 20")
at.segmented_control(key="tr_twin").set_value("Season")
at.segmented_control(key="tr_ms").set_value("Career")
at.run()
print("after controls - exceptions:", [e.value for e in at.exception][:3], "| tables:", [len(d.value) for d in at.dataframe])
at.multiselect(key="tr_games").set_value([at.multiselect(key="tr_games").options[0]])
at.run()
print("one game - exceptions:", [e.value for e in at.exception][:3], "| tables:", [len(d.value) for d in at.dataframe])
