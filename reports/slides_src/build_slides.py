"""Build the result slides (HTML, 1280x720 per slide) in the style of MidTermSlideSimpleDyG.pdf."""
import re
import sys
from pathlib import Path

FIRST_PAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 13
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 29
B = Path('build')


def svg(name, width):
    s = (B / name).read_text(encoding='utf-8')
    s = re.sub(r'<\?xml[^>]*>\s*', '', s)
    s = re.sub(r'<!DOCTYPE[^>]*>\s*', '', s)
    s = re.sub(r'<svg ', f'<svg style="width:{width}px;height:auto;display:block" ', s, count=1)
    s = re.sub(r'(<svg[^>]*?) width="[^"]*"', r'\1', s, count=1)
    s = re.sub(r'(<svg[^>]*?) height="[^"]*"', r'\1', s, count=1)
    return s


CSS = '''
@page { size: 1280px 720px; margin: 0 }
* { box-sizing: border-box; margin: 0; padding: 0 }
body { font-family: "Be Vietnam Pro", "Inter", sans-serif; color: #334155; -webkit-print-color-adjust: exact; print-color-adjust: exact }
.slide { width: 1280px; height: 720px; background: #f8fafc; position: relative; overflow: hidden; padding: 84px 86px 0; page-break-after: always; break-after: page }
.label { font-size: 16.5px; font-weight: 700; color: #0284c7; letter-spacing: .01em; text-transform: uppercase }
h1 { font-size: 41px; line-height: 1.15; font-weight: 700; color: #0f172a; letter-spacing: -.015em; margin: 6px 0 22px }
.footer { position: absolute; left: 86px; right: 86px; bottom: 44px; display: flex; justify-content: space-between; font-size: 16.5px; color: #64748b }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 18px }
.card h3 { font-size: 19px; font-weight: 700; color: #0284c7; margin-bottom: 6px }
.card p, .card li { font-size: 16px; line-height: 1.45; color: #334155 }
.card ul { padding-left: 18px }
.row { display: flex; gap: 16px }
.col { display: flex; flex-direction: column; gap: 12px }
.num { flex: none; width: 34px; height: 34px; border-radius: 50%; background: #0ea5e9; color: #fff; font-weight: 700; font-size: 17px; display: flex; align-items: center; justify-content: center }
.numcard { display: flex; gap: 14px; align-items: flex-start }
.strip { background: #0f172a; color: #e2e8f0; border-radius: 8px; padding: 12px 18px; font-size: 16px; line-height: 1.45 }
.strip b { color: #38bdf8 } .strip .o { color: #fdba74; font-weight: 700 }
.callout { background: #e0f2fe; border: 1px solid #7dd3fc; border-radius: 10px; padding: 14px 18px; font-size: 16px; line-height: 1.45; color: #0f172a }
.callout.warn { background: #fff7ed; border-color: #fdba74 }
.callout .t { font-weight: 700; color: #0284c7; font-size: 13px; text-transform: uppercase; letter-spacing: .02em; margin-bottom: 4px }
.callout.warn .t { color: #c2410c }
.pill { display: inline-block; border-radius: 6px; padding: 2px 9px; font-size: 13px; font-weight: 600; margin: 0 4px 4px 0 }
.pill.b { background: #bae6fd; color: #0c4a6e } .pill.o { background: #fed7aa; color: #9a3412 }
.pill.t { background: #ccfbf1; color: #115e59 } .pill.v { background: #ede9fe; color: #5b21b6 } .pill.g { background: #e2e8f0; color: #334155 }
.pill.ok { background: #dcfce7; color: #166534 } .pill.part { background: #fef3c7; color: #92400e }
table.t { width: 100%; border-collapse: separate; border-spacing: 0; background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; font-size: 16.5px }
table.t th { background: #0f172a; color: #fff; font-weight: 600; text-align: left; padding: 11px 14px; font-size: 16px }
table.t td { padding: 9px 14px; color: #0f172a; border-top: 1px solid #f1f5f9 }
table.t tr:nth-child(even) td { background: #f1f5f9 }
table.t th.hl { background: #7dd3fc; color: #0f172a; font-weight: 700 }
table.t td.hl { background: #bae6fd !important; font-weight: 700 }
table.t td.sub, table.t th.sub { color: #64748b; font-weight: 400 }
table.t th.sub { color: #cbd5e1 }
table.t td.pos { color: #b91c1c } table.t .muted { color: #64748b }
.how { font-size: 14px; color: #475569; margin-top: 8px } .how b { color: #0284c7 }
.note { font-size: 12.5px; color: #64748b; margin-top: 8px; line-height: 1.4 }
.big { font-size: 34px; font-weight: 800; color: #0284c7; line-height: 1 }
.chart { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 12px; display: flex; justify-content: center }
'''


def page(label, title, body, footer, n):
    return (f'<section class="slide"><div class="label">{label}</div><h1>{title}</h1>{body}'
            f'<div class="footer"><span>{footer}</span><span>{n} / {TOTAL}</span></div></section>')


slides = []
n = FIRST_PAGE

# 1a. Setup details
slides.append(page('4. Thực nghiệm · Thiết lập', 'Thiết lập tái lập chi tiết', f"""
<table class="t" style="margin-bottom:14px;font-size:15px"><tr><th>Dataset</th><th># Nút · # Cạnh</th><th>Time step</th><th>Truy vấn train / val / test</th><th>SimpleDyG (theo script tác giả)</th><th>Tham số</th></tr>
<tr><td><b>UCI</b> · Social</td><td>1.781 · 16.743</td><td>13 (T−1 val, T test)</td><td>1.708 / 146 / 110</td><td>6 lớp · 8 head · dim 768 · batch 32 · lr 1e-4</td><td>44,7 triệu</td></tr>
<tr><td><b>ML-10M</b> · Rating</td><td>15.841 · 48.561</td><td>13 (T−1 val, T test)</td><td>13.701 / 738 / 396</td><td>2 lớp · 4 head · dim 128 · batch 64 · lr 1e-5</td><td>2,6 triệu</td></tr></table>
<div class="row" style="margin-bottom:14px">
 <div class="card" style="flex:1.05"><h3>Huấn luyện SimpleDyG</h3><ul>
  <li>AdamW của tác giả, lr giảm tuyến tính, tối đa 100 epoch.</li>
  <li>Early stopping theo NDCG@5 validation, <i>patience</i> 10 (corrected: 20).</li>
  <li>Chuỗi train ≤ 512 token, context 1.024; micro-batch 8 + gradient accumulation, giữ nguyên batch logic.</li>
  <li>Dự đoán: greedy decoding đến token kết thúc.</li></ul></div>
 <div class="card" style="flex:1"><h3>Baseline (DyGLib)</h3><ul>
  <li>TGAT (20 neighbour) và GraphMixer (1.024 token lịch sử).</li>
  <li>Loss BPR, feature nút/cạnh = 0, batch 128.</li>
  <li>Chọn dim ∈ {{16, 32, 64, 128, 256, 512}} theo validation → 6 lần huấn luyện mỗi seed.</li></ul></div>
 <div class="card" style="flex:.95"><h3>Đánh giá</h3><ul>
  <li>NDCG@5 chuẩn (khử trùng lặp) và Jaccard.</li>
  <li>10 seed (0–9), báo cáo mean ± std; so biến thể theo cặp seed, CI 95% (t) và sign test.</li>
  <li>Kiểm tra hash dữ liệu và split trước khi chạy.</li></ul></div>
</div>
<div class="row" style="gap:10px;font-size:15px;align-items:center"><span style="font-weight:700;color:#0f172a">Biến thể temporal token:</span>
 <span><span class="pill b">full</span>gốc</span><span><span class="pill o">same time</span>một token cho mọi slot</span><span><span class="pill t">no time</span>bỏ token</span><span><span class="pill v">per-ego permute</span>xáo nhãn riêng từng ego</span></div>
""", 'SimpleDyG · Thực nghiệm · Thiết lập', n)); n += 1

# 1c. Temporal pattern of the data (Figure A.1)
slides.append(page('4. Thực nghiệm · Dữ liệu theo thời gian', 'Dữ liệu thay đổi mạnh theo thời gian', f"""
<div class="chart">{svg('ch_steps.svg', 1080)}</div>
<div class="row" style="margin-top:10px">
 <div class="card numcard" style="flex:1;padding:12px 16px"><div class="num">1</div><div><h3>UCI thưa dần</h3><p>Từ ~3.600 tương tác/step còn <b>197</b> ở step test (<b>110 user</b>) → tập test nhỏ, điểm dao động giữa seed.</p></div></div>
 <div class="card numcard" style="flex:1;padding:12px 16px"><div class="num">2</div><div><h3>ML-10M ổn định hơn</h3><p>~3.000–4.000 tương tác/step, có đỉnh ở step 0 và 6; step test còn 991 tương tác của 396 user.</p></div></div>
 <div class="card numcard" style="flex:1;padding:12px 16px"><div class="num">3</div><div><h3>Lập luận của paper</h3><p>Paper (Hình A.1) dùng các thay đổi đột ngột này để biện minh cho temporal alignment; slide 23 kiểm tra lại.</p></div></div>
</div>
""", 'SimpleDyG · Thực nghiệm · Dữ liệu (Hình A.1)', n)); n += 1

# 1b. Infrastructure and cost
slides.append(page('4. Thực nghiệm · Hạ tầng và chi phí', 'Chi phí chạy trên Kaggle 2×T4', f"""
<div class="row">
 <div class="col" style="flex:1">
  <table class="t" style="font-size:15px"><tr><th>Đợt</th><th>Nội dung</th><th>Lượt</th><th style="white-space:nowrap">GPU-giờ</th><th class="hl" style="white-space:nowrap">2×T4</th></tr>
  <tr><td><b>1</b></td><td>2 dataset × 5 cấu hình × 10 seed</td><td>100</td><td>670</td><td class="hl">≈ 335 giờ</td></tr>
  <tr><td><b>2</b></td><td>UCI per-ego permute · 10 seed</td><td>10</td><td>6,4</td><td class="hl">≈ 3,2 giờ</td></tr>
  <tr><td><b>3</b></td><td>ML-10M corrected · 3 × 5 seed</td><td>15</td><td>12,6</td><td class="hl">≈ 6,5 giờ</td></tr>
  <tr><td><b>Tổng</b></td><td></td><td><b>125</b></td><td><b>≈ 689</b></td><td class="hl">≈ 345 giờ</td></tr></table>
  <p class="note" style="margin-top:6px">GPU-giờ = tổng thời gian huấn luyện và test ghi trong log, gồm cả decode validation mỗi epoch. Thời gian 2×T4 = GPU-giờ / 2, chưa gồm cài đặt và nạp lại checkpoint giữa các phiên.</p>
  <div class="card" style="margin-top:2px"><h3>Hạ tầng</h3><p>Kaggle 2 × Tesla T4 (16 GB), PyTorch 2.11 / CUDA 12.8. Hai worker độc lập, mỗi GPU một lượt; checkpoint 5 phút/lần, phiên ≤ 11,75 giờ rồi chạy tiếp.</p></div>
 </div>
 <div class="col" style="flex:none;width:470px">
  <div class="chart">{svg('ch_cost.svg', 446)}</div>
  <div class="row">
   <div class="card" style="flex:1;padding:12px 14px"><div class="big" style="font-size:28px">&lt; 10%</div><p>tổng GPU-giờ dành cho SimpleDyG (≈ 66 / 689).</p></div>
   <div class="card" style="flex:1;padding:12px 14px"><div class="big" style="font-size:28px">47 s</div><p>/epoch SimpleDyG trên UCI, so với 67 s (TGAT) và 259 s (GraphMixer).</p></div>
  </div>
 </div>
</div>
""", 'SimpleDyG · Thực nghiệm · Chi phí', n)); n += 1

# 2. Table 2
slides.append(page('4. Thực nghiệm · Tái lập Table 2', 'Kết quả so với paper', '''
<table class="t"><tr><th>Tập · Metric</th><th>TGAT</th><th class="sub">paper</th><th>GraphMixer</th><th class="sub">paper</th><th class="hl">SimpleDyG (chúng tôi)</th><th class="sub">paper</th></tr>
<tr><td>UCI · NDCG@5</td><td>.049 ± .024</td><td class="sub">.061</td><td>.070 ± .017</td><td class="sub">.104</td><td class="hl">.104 ± .017</td><td class="sub">.104</td></tr>
<tr><td>UCI · Jaccard</td><td>.028 ± .013</td><td class="sub">.020</td><td>.034 ± .009</td><td class="sub">.042</td><td class="hl">.087 ± .013</td><td class="sub">.092</td></tr>
<tr><td>ML-10M · NDCG@5</td><td>.009 ± .014</td><td class="sub">.066</td><td>.152 ± .046</td><td class="sub">.081</td><td class="hl">.099 ± .061</td><td class="sub">.138</td></tr>
<tr><td>ML-10M · Jaccard</td><td>.005 ± .006</td><td class="sub">.021</td><td>.086 ± .046</td><td class="sub">.043</td><td class="hl">.093 ± .057</td><td class="sub">.131</td></tr></table>
<p class="note">Mean ± std trên 10 seed, giao thức author_compat. Jaccard của baseline = max theo k ∈ {1, 5, 10, 20} trên test như paper; SimpleDyG dùng tập sinh ra. Chỉ chạy 2/8 baseline của paper.</p>
<div class="row" style="margin-top:16px">
 <div class="callout" style="flex:1"><div class="t">✓ UCI · tái lập được</div>SimpleDyG đạt <b>.104</b>, đúng bằng paper, và tốt nhất trong ba mô hình ở cả hai metric. Std giữa seed (.017) tương đương paper.</div>
 <div class="callout warn" style="flex:1"><div class="t">✗ ML-10M · lệch so với paper</div>SimpleDyG thấp hơn paper <b>.039</b>, std lớn gấp ~6 lần; GraphMixer lại "vượt" paper. Nguyên nhân: mô hình <b>suy biến</b> về dự đoán phổ biến (slide 19).</div>
</div>
''', 'SimpleDyG · Thực nghiệm · Table 2', n)); n += 1

# 3. Visual comparison
slides.append(page('4. Thực nghiệm · So sánh trực quan', 'Mô hình, paper và baseline không huấn luyện', f'''
<div class="chart">{svg('ch_table2.svg', 1080)}</div>
<div class="strip" style="margin-top:14px"><b>UCI:</b> heuristic "lặp lại slot gần nhất" (<span class="o">.293</span>) cao gấp ~2,8 lần mọi mô hình. &nbsp; <b>ML-10M:</b> không mô hình nào vượt "luôn đoán phim phổ biến nhất" (<span class="o">.148</span>).</div>
''', 'SimpleDyG · Thực nghiệm · Hình: NDCG@5, ◇ = paper', n)); n += 1

# 4. Trivial baselines
slides.append(page('Phát hiện 1 · Baseline đơn giản', 'Heuristic không huấn luyện mạnh hơn mô hình', f'''
<div class="chart">{svg('ch_trivial.svg', 1080)}</div>
<div class="row" style="margin-top:12px">
 <div class="card" style="flex:1;display:flex;gap:14px;align-items:center;padding:12px 16px"><div class="big">2,8×</div><p>"Lặp lại slot gần nhất" so với SimpleDyG trên UCI (.293 vs .104); vẫn giữ trên validation (.219).</p></div>
 <div class="card" style="flex:1;padding:12px 16px"><h3>Vì sao?</h3><p>UCI có tính lặp lại rất mạnh: người dùng nhắn lại đúng những người vừa nhắn ở slot trước.</p></div>
 <div class="card" style="flex:1;padding:12px 16px"><h3>Hàm ý</h3><p>Paper không có baseline này; cần luôn báo cáo kèm khi đánh giá dynamic link prediction.</p></div>
</div>
''', 'SimpleDyG · Thực nghiệm · Baseline không huấn luyện', n)); n += 1

# 5. Collapse
slides.append(page('Phát hiện 2 · ML-10M suy biến', 'ML-10M: mô hình chỉ đoán một bộ phim', f'''
<div class="row">
 <div class="chart" style="flex:none;width:640px">{svg('ch_collapse.svg', 610)}</div>
 <div class="col" style="flex:1">
  <div class="card numcard"><div class="num">1</div><div><h3>Phim 1419</h3><p>Đích phổ biến nhất khi train (1.417 lần), có trong ground truth của 17,7% user test. 23/30 lượt mã gốc đoán phim này cho 75–100% user.</p></div></div>
  <div class="card numcard"><div class="num">2</div><div><h3>Trần điểm = độ phổ biến</h3><p>Luôn đoán 1419 cho NDCG@5 <b>.148</b> / Jaccard <b>.140</b>, đúng mức cao nhất của mọi lượt SimpleDyG và GraphMixer.</p></div></div>
  <div class="card numcard"><div class="num">3</div><div><h3>Trùng với paper</h3><p>Paper báo ML-10M same time = <b>.147 ± .001</b>, gần như đúng mức này (suy luận từ số liệu).</p></div></div>
 </div>
</div>
<div class="strip" style="margin-top:14px">UCI: <b>63–96</b> tập dự đoán khác nhau trên 110 truy vấn → dự đoán cá nhân hoá. &nbsp; ML-10M: chỉ <span class="o">2–9</span> tập trên 396 truy vấn.</div>
''', 'SimpleDyG · Thực nghiệm · Kiểm tra suy biến', n)); n += 1

# 6. ML fix
slides.append(page('Chẩn đoán · ML-10M', 'Mask padding: học sớm hơn nhưng vẫn suy biến', f'''
<div class="row">
 <div class="chart" style="flex:none;width:610px">{svg('ch_mlfix.svg', 584)}</div>
 <div class="col" style="flex:1">
  <div class="card"><h3>Giả thuyết</h3><p>Loss tính cả trên padding, lr 1e-5: NDCG val = 0 suốt ~20 epoch → <i>patience</i> 10 dừng sớm.</p></div>
  <div class="card"><h3>Thử nghiệm: protocol corrected</h3><p>Mask padding + attention mask, <i>patience</i> 20, 3 biến thể × 5 seed.</p></div>
  <div class="card"><h3>Kết quả</h3><ul><li>full ổn định hơn: std <b>.061 → .011</b>, không còn seed = 0.</li><li>Nhưng .120 &lt; .148, <b>14/15</b> lượt vẫn suy biến; same time: 5/5 = 0.</li></ul></div>
 </div>
</div>
<div class="strip" style="margin-top:14px">Kết luận: với cấu hình paper (2 lớp, dim 128, lr 1e-5), ML-10M <span class="o">bị chi phối bởi độ phổ biến</span> → không dùng để kết luận về temporal alignment.</div>
''', 'SimpleDyG · Thực nghiệm · Chẩn đoán ML-10M', n)); n += 1

# 6b. Validation curves: UCI
slides.append(page('Chẩn đoán · Quá trình học (1/2)', 'UCI: mô hình học bình thường', f"""
<div class="row">
 <div class="chart" style="flex:none;width:690px">{svg('ch_curves1.svg', 664)}</div>
 <div class="col" style="flex:1;gap:10px">
  <div class="card numcard" style="padding:12px 14px"><div class="num">1</div><div><h3>Học ngay từ đầu</h3><p>Điểm rời 0 từ epoch ~4 và tăng đều, không có giai đoạn "đứng yên".</p></div></div>
  <div class="card numcard" style="padding:12px 14px"><div class="num">2</div><div><h3>Bão hoà ở .15–.18</h3><p>Sau ~30 epoch điểm chững lại; early stopping dừng ở epoch 20–58.</p></div></div>
  <div class="card numcard" style="padding:12px 14px"><div class="num">3</div><div><h3>Ba biến thể chồng nhau</h3><p>Ba biến thể học gần như giống hệt nhau → temporal token không đổi cách học.</p></div></div>
 </div>
</div>
""", 'SimpleDyG · Chẩn đoán · Đường validation UCI', n)); n += 1

# 6c. Validation curves: ML-10M
slides.append(page('Chẩn đoán · Quá trình học (2/2)', 'ML-10M: kẹt ở độ phổ biến, kể cả khi mask padding', f"""
<div class="chart">{svg('ch_curves2.svg', 1080)}</div>
<div class="row" style="margin-top:14px">
 <div class="card numcard" style="flex:1;padding:12px 16px"><div class="num">1</div><div><h3>Mã gốc: 0 rồi nhảy vọt</h3><p>Điểm = 0 suốt ~22–25 epoch, rồi nhảy lên <b>đúng .0955</b> và đứng yên (20/30 lượt). full rời đỉnh thì tụt về ~.03.</p></div></div>
 <div class="card numcard" style="flex:1;padding:12px 16px"><div class="num">2</div><div><h3>Mask padding: sớm hơn, cùng đích</h3><p>no time học từ epoch <b>4</b> (thay vì ~22) nhưng vẫn dừng ở .0955; full chỉ sớm hơn ít; same time = 0.</p></div></div>
</div>
""", 'SimpleDyG · Chẩn đoán · Đường validation ML-10M', n)); n += 1

# 7. Temporal alignment chart
slides.append(page('4. Thực nghiệm · Temporal alignment', 'Vai trò của temporal alignment trên UCI', f'''
<div class="chart">{svg('ch_temporal.svg', 1080)}</div>
<div class="row" style="margin-top:8px;font-size:14px;color:#334155;gap:22px">
 <div><span class="pill o">same time</span>một token cho mọi slot</div><div><span class="pill t">no time</span>bỏ token, giữ thứ tự sự kiện</div>
 <div><span class="pill v">per-ego permute</span>giữ phân đoạn, xáo nhãn riêng từng ego</div></div>
<div class="strip" style="margin-top:6px;padding:10px 18px">Không biến thể nào khác <b>full</b> có ý nghĩa thống kê (mọi CI 95% chứa 0) → trên UCI, temporal token và căn chỉnh thời gian giữa các ego <span class="o">không có tác dụng đo được</span>.</div>
''', 'SimpleDyG · Thực nghiệm · Temporal alignment', n)); n += 1

# 8. Table 4
slides.append(page('4. Thực nghiệm · Tái lập Table 4', 'Temporal alignment: chúng tôi so với paper', '''
<table class="t"><tr><th>Biến thể</th><th class="hl">UCI (chúng tôi)</th><th class="sub">paper</th><th>Δ so với full [CI 95%]</th><th>ML-10M (mã gốc)</th><th class="sub">paper</th></tr>
<tr><td><span class="pill b">full</span></td><td class="hl">.104 ± .017</td><td class="sub">.104</td><td class="muted">–</td><td>.099 ± .061</td><td class="sub">.138</td></tr>
<tr><td><span class="pill o">same time</span></td><td class="hl">.104 ± .017</td><td class="sub">.090</td><td>−.001 [−.016, +.015]</td><td>.118 ± .062</td><td class="sub">.147</td></tr>
<tr><td><span class="pill t">no time</span></td><td class="hl">.109 ± .011</td><td class="sub">.111</td><td>+.005 [−.009, +.018]</td><td>.118 ± .062</td><td class="sub">.117</td></tr>
<tr><td><span class="pill v">per-ego permute</span></td><td class="hl">.097 ± .022</td><td class="sub">–</td><td>−.007 [−.029, +.014]</td><td class="muted">không chạy</td><td class="sub">–</td></tr></table>
<p class="note">NDCG@5 chuẩn, mean ± std trên 10 seed. Δ so cặp cùng seed; sign test p ≥ .75 cho cả ba so sánh trên UCI.</p>
<div class="row" style="margin-top:16px">
 <div class="callout" style="flex:1"><div class="t">Paper kết luận</div>Temporal alignment quan trọng với UCI và ML-10M; same time kém nhất trên UCI (.090).</div>
 <div class="callout warn" style="flex:1"><div class="t">Chúng tôi thấy</div>UCI: thứ tự trung bình giống paper (no time ≥ full ≥ same time) nhưng <b>nằm trong nhiễu seed</b>. ML-10M: so sánh không hợp lệ vì mọi biến thể suy biến.</div>
</div>
''', 'SimpleDyG · Thực nghiệm · Table 4', n)); n += 1

# 9a/9b. Supplementary analyses (formerly appendix)
slides.append(page('Phân tích bổ sung · So cặp theo seed', 'Từng seed thay đổi thế nào giữa các biến thể?', f"""
<div class="chart">{svg('ch_paired.svg', 1060)}</div>
<div class="row" style="margin-top:8px">
 <div class="callout" style="flex:1;padding:10px 16px"><div class="t">UCI (hàng trên)</div>Các đường đan chéo nhau, số seed tăng ≈ số seed giảm → chênh lệch là nhiễu giữa seed, khớp với CI 95% chứa 0 ở slide 23.</div>
 <div class="callout warn" style="flex:1;padding:10px 16px"><div class="t">ML-10M (hàng dưới)</div>Seed chỉ nhảy giữa hai mức: <b>0</b> (không sinh dự đoán) và <b>.147</b> (đoán phim 1419) → so biến thể = đếm seed rơi vào mức nào.</div>
</div>
""", 'SimpleDyG · Phân tích bổ sung · So cặp seed', n)); n += 1
slides.append(page('Phân tích bổ sung · Hidden dimension của baseline', 'Chọn dimension: tốn kém nhưng ít tác dụng', f"""
<div class="chart">{svg('ch_dims.svg', 1080)}</div>
<div class="row" style="margin-top:10px">
 <div class="card" style="flex:1;padding:12px 16px"><h3>Gần như phẳng</h3><p>Khác biệt giữa các dimension phần lớn nằm trong sai số giữa seed; dim lớn không cho kết quả tốt hơn rõ rệt.</p></div>
 <div class="card" style="flex:1;padding:12px 16px"><h3>Nhưng rất tốn</h3><p>GraphMixer ML-10M dim 512 mất <b>24,5 giờ</b>/seed (dim 16: 1,8 giờ) — hơn một nửa trong 42 giờ/lượt ở slide 15.</p></div>
 <div class="card" style="flex:1;padding:12px 16px"><h3>Validation ≠ test</h3><p>UCI: baseline chỉ đạt .015–.034 trên validation nhưng .05–.07 trên test → chọn theo validation khá nhiễu.</p></div>
</div>
""", 'SimpleDyG · Phân tích bổ sung · Hidden dimension', n)); n += 1

# 10. Summary of results
slides.append(page('Tổng kết · Kết quả đạt được', 'Tổng kết các kết quả đã chạy', """
<table class="t" style="font-size:15.5px"><tr><th style="width:21%">Hạng mục</th><th>Kết quả</th><th style="width:21%">Đánh giá</th></tr>
<tr><td><b>Dữ liệu</b></td><td>UCI và ML-10M từ bản preprocess của tác giả; kiểm tra hash, split và khớp với đồ thị gốc. 125/125 lượt chạy hoàn tất, ≈ 689 GPU-giờ.</td><td><span class="pill ok">Hoàn tất</span></td></tr>
<tr><td><b>Link prediction · UCI</b></td><td>SimpleDyG <b>.104 ± .017</b> = paper (.104), tốt hơn TGAT (.049) và GraphMixer (.070).</td><td><span class="pill ok">Khớp paper</span></td></tr>
<tr><td><b>Link prediction · ML-10M</b></td><td>.099 ± .061 so với paper .138. Mô hình suy biến về phim phổ biến nhất (trần .148); mask padding giúp ổn định (std .011) nhưng không khắc phục được.</td><td><span class="pill part">Lệch · đã tìm nguyên nhân</span></td></tr>
<tr><td><b>Baseline kiểm chứng</b></td><td>Heuristic "lặp lại slot gần nhất" đạt <b>.293</b> trên UCI, ~2,8× mọi mô hình.</td><td><span class="pill part">Phát hiện mới</span></td></tr>
<tr><td><b>Temporal alignment</b></td><td>4 biến thể × 10 seed trên UCI; Δ so với full: same time −.001, no time +.005, per-ego permute −.007, mọi CI 95% chứa 0.</td><td><span class="pill g">Không có hiệu ứng đo được</span></td></tr></table>
<div class="strip" style="margin-top:16px"><b>Tóm lại:</b> tái lập được kết quả chính trên UCI; kết quả ML-10M và vai trò của temporal alignment <span class="o">không được xác nhận</span> trong phạm vi thực nghiệm này.</div>
""", 'SimpleDyG · Tổng kết', n)); n += 1

# 11. Comments, limitations, future work
slides.append(page('Nhận xét · Đồ án', 'Nhận xét, hạn chế và hướng phát triển', """
<div class="row" style="margin-bottom:14px">
 <div class="card numcard" style="flex:1"><div class="num">1</div><div><h3>Đơn giản và rẻ</h3><p>Transformer gốc tái lập được trên UCI, chỉ tốn &lt; 10% GPU-giờ so với baseline; GraphMixer ML-10M mất 42 giờ/lượt.</p></div></div>
 <div class="card numcard" style="flex:1"><div class="num">2</div><div><h3>Đánh giá cần baseline đơn giản</h3><p>Heuristic không huấn luyện vượt mọi mô hình trên UCI; kết luận "vượt trội" của paper cần đặt cạnh baseline này.</p></div></div>
 <div class="card numcard" style="flex:1"><div class="num">3</div><div><h3>Temporal token chưa quyết định</h3><p>Bỏ hoặc xáo token thời gian không làm kết quả UCI thay đổi đáng kể; tín hiệu chính là tính lặp lại của từng ego.</p></div></div>
 <div class="card numcard" style="flex:1"><div class="num">4</div><div><h3>Cần kiểm tra suy biến</h3><p>Trên ML-10M, điểm số phản ánh độ phổ biến của phim, kể cả số của paper (same time .147 ± .001).</p></div></div>
</div>
<div class="row">
 <div class="callout warn" style="flex:1"><div class="t">Hạn chế</div><ul style="padding-left:18px"><li>2/4 dataset, 2/8 baseline (adapter BPR, feature 0).</li><li>10 seed chỉ phát hiện được hiệu ứng ≳ .015 NDCG; ML-10M corrected mới chạy 5 seed.</li><li>Nhận định về số ML-10M của paper là suy luận, chưa kiểm chứng bằng checkpoint của tác giả.</li></ul></div>
 <div class="callout" style="flex:1"><div class="t">Hướng phát triển</div><ul style="padding-left:18px"><li>Đưa baseline "lặp lại / phổ biến" thành chuẩn khi đánh giá.</li><li>ML-10M: tăng lr hoặc kích thước mô hình, phạt node phổ biến khi decode.</li><li>Thêm MMConv, Hepth; thử alignment thích nghi theo ego.</li></ul></div>
</div>
""", 'SimpleDyG · Nhận xét', n)); n += 1

html = '<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8"><title>SimpleDyG – Kết quả thực nghiệm</title><style>' + CSS + '</style></head><body>' + ''.join(slides) + '</body></html>'
(B / 'results_slides.html').write_text(html, encoding='utf-8')
print('slides', len(slides), 'pages', FIRST_PAGE, '-', n - 1, 'of', TOTAL)
