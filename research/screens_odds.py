"""Game page with the odds switch on, and the player page."""
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path(__file__).parent / "shots"
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1600, "height": 2400})
    pg.goto("http://localhost:8670/game"); pg.wait_for_selector("h1", timeout=120000); pg.wait_for_timeout(8000)
    pg.screenshot(path=str(OUT / "game_pct.png"))
    pg.get_by_text("Fair odds", exact=True).click(); pg.wait_for_timeout(6000)
    pg.screenshot(path=str(OUT / "game_odds.png"))
    pg.goto("http://localhost:8670/player"); pg.wait_for_selector("h1", timeout=120000); pg.wait_for_timeout(9000)
    pg.screenshot(path=str(OUT / "player.png"))
    b.close()
