from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path(__file__).parent / "shots"; OUT.mkdir(exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1600, "height": 1300})
    pg.goto("http://localhost:8670/"); pg.wait_for_selector("h1", timeout=120000); pg.wait_for_timeout(9000)
    pg.screenshot(path=str(OUT / "slate_pct.png"))
    pg.get_by_text("Fair odds", exact=True).click(); pg.wait_for_timeout(6000)
    pg.screenshot(path=str(OUT / "slate_odds.png"))
    pg.goto("http://localhost:8670/game"); pg.wait_for_selector("h1", timeout=120000); pg.wait_for_timeout(9000)
    pg.mouse.wheel(0, 900); pg.wait_for_timeout(1500)
    pg.screenshot(path=str(OUT / "game_tbl.png"))
    b.close()
