from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path(__file__).parent / "shots"
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1600, "height": 1100})
    pg.goto("http://localhost:8670/"); pg.wait_for_selector("h1", timeout=120000); pg.wait_for_timeout(9000)
    grid = pg.locator("[data-testid='stDataFrame']").last
    box = grid.bounding_box(); pg.mouse.move(box["x"] + box["width"] / 2, box["y"] + 120)
    pg.mouse.wheel(1200, 0); pg.wait_for_timeout(2000)
    pg.screenshot(path=str(OUT / "slate_scrolled.png"))
    b.close()
