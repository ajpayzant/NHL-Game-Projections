"""Screenshots of app pages through the installed Edge (python -m research.screens [page ...])."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path(__file__).parent / "shots"; OUT.mkdir(exist_ok=True)
pages = sys.argv[1:] or ["slate", "game", "lines", "player", "model"]
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1600, "height": 3400})
    for name in pages:
        pg.goto("http://localhost:8670/" + ("" if name == "slate" else name))
        pg.wait_for_selector("h1", timeout=120000)
        pg.wait_for_timeout(9000)
        pg.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
        print(name, "ok")
    b.close()
