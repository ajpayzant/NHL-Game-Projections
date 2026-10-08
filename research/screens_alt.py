"""Browser check of the total arrows on the Game board: python -m research.screens_alt"""
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path(__file__).parent / "shots"; OUT.mkdir(exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1400, "height": 1200})
    pg.goto("http://localhost:8670/game")
    pg.wait_for_selector(".gb .ar b", timeout=180000)
    pg.wait_for_timeout(3000)
    read = lambda: [pg.inner_text(f"[id$='-{s}']").replace("\n", " ") for s in ("o", "u")]
    print("start  ", read())
    board = pg.locator(".gb").first
    board.screenshot(path=str(OUT / "alt_board_start.png"))
    up = pg.locator(".gb .ar b[data-d='1']").first
    down = pg.locator(".gb .ar b[data-d='-1']").first
    up.click(); print("up 1   ", read())
    up.click(); print("up 2   ", read())
    board.screenshot(path=str(OUT / "alt_board_up2.png"))
    for _ in range(4):
        down.click()
    print("down 4 ", read())
    for _ in range(10):
        down.click()
    print("floor  ", read())
    b.close()
