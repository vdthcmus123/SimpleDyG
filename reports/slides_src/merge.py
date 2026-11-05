"""Rebuild SimpleDyG_Slide_KetQua.pdf.

Steps (run from this folder; needs pandas, matplotlib + Inter font, playwright with Chrome, pypdf):
  python prepare_data.py
  python charts.py          # needs data/all_runs.csv and the two paired_temporal_deltas.csv files
  python build_slides.py 13 29
  python render.py $PWD/build/results_slides.html $PWD/build/results_slides.pdf
  python merge.py MidTermSlideSimpleDyG.pdf build/results_slides.pdf ../SimpleDyG_Slide_KetQua.pdf
Original pages 2-12 get their baked-in "n / 13" covered and rewritten as "n / TOTAL".
"""
import subprocess
import sys
from pathlib import Path
from pypdf import PdfReader, PdfWriter

orig_path, new_path, out_path = sys.argv[1:4]
appendix = PdfReader(sys.argv[4]) if len(sys.argv) > 4 else None   # optional, placed after "Thank you"
orig, new = PdfReader(orig_path), PdfReader(new_path)
keep_front = len(orig.pages) - 1            # every original page except the closing "Thank you"
total = keep_front + len(new.pages) + 1 + (len(appendix.pages) if appendix else 0)

pages = ''.join(f'<section class="s"><div class="cover"></div><div class="num">{n} / {total}</div></section>'
                for n in range(2, keep_front + 1))
html = ('<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
        '@page{size:1280px 720px;margin:0}*{margin:0;padding:0;box-sizing:border-box}html,body{background:transparent}'
        '.s{width:1280px;height:720px;position:relative;page-break-after:always;break-after:page}'
        '.cover{position:absolute;left:1118px;top:652px;width:86px;height:28px;background:#f8fafc}'
        '.num{position:absolute;right:86px;bottom:44px;font-family:"Be Vietnam Pro","Inter",sans-serif;font-size:16.5px;color:#64748b}'
        '</style></head><body>' + pages + '</body></html>')
Path('build').mkdir(exist_ok=True)
Path('build/overlay.html').write_text(html, encoding='utf-8')
subprocess.run([sys.executable, 'render.py', str(Path('build/overlay.html').resolve()), str(Path('build/overlay.pdf').resolve())], check=True)
overlay = PdfReader('build/overlay.pdf')

writer = PdfWriter()
for i in range(keep_front):
    page = orig.pages[i]
    if i >= 1:
        page.merge_page(overlay.pages[i - 1])
    writer.add_page(page)
for page in new.pages:
    writer.add_page(page)
writer.add_page(orig.pages[-1])
if appendix:
    for page in appendix.pages:
        writer.add_page(page)
writer.add_metadata({'/Title': 'SimpleDyG – Giới thiệu, phương pháp và kết quả thực nghiệm'})
with open(out_path, 'wb') as f:
    writer.write(f)
print(out_path, len(writer.pages), 'pages')
