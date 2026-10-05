"""Headless check of the goalie tables: python -m research.apptest_goalies"""
import os
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
os.environ["APP_PAGE"] = "slate"
at = AppTest.from_file(APP, default_timeout=600)
at.run()
at.segmented_control(key="slate_kind").set_value("Goalies").run()
print("slate/goalies exceptions:", [e.value for e in at.exception][:2], "| tables:", len(at.dataframe))
df = at.dataframe[-1].value
print(df.head(20).to_string())
at.toggle(key="g_backups").set_value(True).run()
print("with backups:", len(at.dataframe[-1].value), "rows")
at.segmented_control(key="prob_view").set_value("Fair odds").run()
print(at.dataframe[-1].value.iloc[:3, -9:].to_string())
os.environ["APP_PAGE"] = "game"
at = AppTest.from_file(APP, default_timeout=600)
at.run()
print("game exceptions:", [e.value for e in at.exception][:2], "| tables:", len(at.dataframe))
print(at.dataframe[2].value.to_string())
