"""Screenshots of every Trends tab (python -m research.screens_trends [port]); the app must be running."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path(__file__).parent / "shots"; OUT.mkdir(exist_ok=True)
port = sys.argv[1] if len(sys.argv) > 1 else "8670"
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1600, "height": 1800})
    pg.goto(f"http://localhost:{port}/trends")
    pg.wait_for_selector("h1", timeout=180000)
    pg.wait_for_timeout(15000)
    for i, tab in enumerate(["Hot & cold", "Streaks", "Matchups", "Milestones", "Hit rates", "Teams"]):
        pg.get_by_role("tab", name=tab).click()
        pg.wait_for_timeout(4000)
        pg.screenshot(path=str(OUT / f"trends_{i}.png"), full_page=True)
        print(tab, "ok")
    b.close()
