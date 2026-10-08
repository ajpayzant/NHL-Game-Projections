"""Browser check of the alternate-line arrows (total and puck line) on the Game board:
python -m research.screens_alt"""
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path(__file__).parent / "shots"; OUT.mkdir(exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1400, "height": 1200})
    pg.goto("http://localhost:8670/game")
    pg.wait_for_selector(".gb .ar b", timeout=180000)
    pg.wait_for_timeout(3000)
    cell = lambda k: pg.inner_text(f"[id$='-{k}']").replace("\n", " ").replace(" ▲ ▼", "")
    read = lambda: f"total [{cell('to')} | {cell('tu')}]  puck [{cell('pf')} | {cell('pd')}]"
    arrow = lambda g, d: pg.locator(f".gb .ar[data-g='{g}'] b[data-d='{d}']").first
    print("start        ", read())
    arrow("p", 1).click(); print("puck up      ", read())
    arrow("p", 1).click(); print("puck up      ", read())
    arrow("p", 1).click(); print("puck up (cap)", read())
    pg.locator(".gb").first.screenshot(path=str(OUT / "alt_board_puck.png"))
    arrow("t", 1).click(); print("total up     ", read())
    arrow("p", -1).click(); arrow("p", -1).click(); arrow("p", -1).click(); print("puck down x3 ", read())
    b.close()
