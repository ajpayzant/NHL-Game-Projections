"""Browser check of the total arrows on the Slate's game cards."""
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path(__file__).parent / "shots"
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1500, "height": 1400})
    pg.goto("http://localhost:8670/")
    pg.wait_for_selector("h1", timeout=180000)
    pg.wait_for_timeout(6000)
    pg.get_by_text("Game lines (fair moneyline, puck line, total)").click()
    pg.wait_for_selector(".gb.sm .ar b", timeout=60000)
    pg.wait_for_timeout(1500)
    n = pg.locator(".gb.sm").count()
    second = pg.locator(".gb.sm").nth(1)
    rd = lambda c: c.locator("[id$='-pf']").inner_text().replace("\n", " ")
    before = [rd(pg.locator(".gb.sm").nth(i)) for i in range(2)]
    second.locator(".ar[data-g='p'] b[data-d='1']").first.click()
    second.locator(".ar[data-g='p'] b[data-d='1']").first.click()
    after = [rd(pg.locator(".gb.sm").nth(i)) for i in range(2)]
    print(f"{n} cards | card 1 {before[0]} -> {after[0]} (untouched) | card 2 {before[1]} -> {after[1]}")
    pg.locator(".gb.sm").nth(0).screenshot(path=str(OUT / "alt_slate_card.png"))
    b.close()
