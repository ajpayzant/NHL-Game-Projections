"""Headless check of the Edges tab on OpticOdds prices: python -m research.apptest_optic"""
import os
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
os.environ["APP_PAGE"] = "edges"
at = AppTest.from_file(APP, default_timeout=300)
at.run()
print("exceptions:", [e.value for e in at.exception][:2], "| errors:", [e.value for e in at.error])
print("tables:", len(at.dataframe), "| metrics:", [(m.label, m.value) for m in at.metric])
for c in at.caption:
    if "Matched" in c.value or "prices" in c.value:
        print("caption:", c.value[:300])
if len(at.dataframe):
    df = at.dataframe[0].value
    print(len(df), "rows"); print(df.head(12).to_string())
    for col in ("Bet", "Type"):
        if col in df:
            print(df[col].astype(str).str.split().str[-1].value_counts().head(10))
for el in list(at.caption) + list(at.markdown) + list(at.info):
    if "Matched" in el.value:
        print("match:", el.value[el.value.find("Matched"):])
