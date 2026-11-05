import sys, asyncio
from playwright.sync_api import sync_playwright
src, out = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    import os, shutil
    chrome = os.environ.get('CHROME') or shutil.which('google-chrome') or shutil.which('chrome')
    b = p.chromium.launch(executable_path=chrome, args=['--no-sandbox']) if chrome else p.chromium.launch()
    pg = b.new_page(viewport={'width': 1280, 'height': 720})
    pg.goto('file://' + src); pg.wait_for_timeout(500)
    # overflow check (prints per-slide content bottom vs footer top)
    print(pg.evaluate("""() => [...document.querySelectorAll('.slide')].map((s,i)=>{const f=s.querySelector('.footer').getBoundingClientRect().top; let m=0; s.querySelectorAll('.chart,.strip,table,.card,.callout,.row,.note').forEach(e=>{m=Math.max(m,e.getBoundingClientRect().bottom)}); return [i+13, Math.round(m - s.getBoundingClientRect().top), Math.round(f - s.getBoundingClientRect().top)]})"""))
    pg.pdf(path=out, width='1280px', height='720px', print_background=True)
    b.close()
