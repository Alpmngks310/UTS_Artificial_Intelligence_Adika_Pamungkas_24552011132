from pathlib import Path
import json
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "laporan" / "laporan_uts_ai_kasus06.pdf"
META = json.loads((ROOT / "output" / "metrics.json").read_text(encoding="utf-8"))
CV = pd.read_csv(ROOT / "output" / "perbandingan_cv.csv")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="MainTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=16, leading=19, alignment=TA_CENTER, textColor=colors.HexColor("#15324B"), spaceAfter=5))
styles.add(ParagraphStyle(name="SubTitle2", parent=styles["Normal"], fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#52616B"), spaceAfter=8))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=11, leading=14, textColor=colors.HexColor("#15324B"), spaceBefore=7, spaceAfter=4))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=8.4, leading=11, spaceAfter=4))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.4, leading=9, textColor=colors.HexColor("#475569"), spaceAfter=2))
styles.add(ParagraphStyle(name="Cell", parent=styles["BodyText"], fontSize=7.4, leading=9))
styles.add(ParagraphStyle(name="CellHead", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=7.4, leading=9, textColor=colors.white))

story=[]
def P(t, s="Bodyx"): story.append(Paragraph(t, styles[s]))
def make_table(data, widths, header=True):
    t=Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    cmds=[("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#CBD5E1")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]
    if header: cmds += [("BACKGROUND",(0,0),(-1,0),colors.HexColor("#15324B")),("TEXTCOLOR",(0,0),(-1,0),colors.white)]
    for r in range(1 if header else 0,len(data)):
        if r%2==0: cmds.append(("BACKGROUND",(0,r),(-1,r),colors.HexColor("#F1F5F9")))
    t.setStyle(TableStyle(cmds)); return t

P("LAPORAN UJIAN TENGAH SEMESTER", "MainTitle")
P("INTELEGENSI BUATAN - MACHINE LEARNING: KLASIFIKASI", "SubTitle2")
P("Kasus 6: Prediksi Negara Asal Mobil (Car Origin Classification)", "H2x")
P("<b>Nama:</b> .................................................... &nbsp;&nbsp; <b>NIM:</b> 24552011132 &nbsp;&nbsp; <b>Kelas:</b> TIF RP 23D", "Bodyx")
P("<b>Penentuan kasus:</b> jumlah digit NIM = 26; 26 mod 10 = 6, sehingga dataset yang digunakan adalah <i>kasus06_cars.csv</i> dengan target <i>Origin</i> (USA, Europe, Japan).", "Bodyx")
P("1. Pemahaman dan Audit Dataset", "H2x")
P("Dataset berisi 406 baris dan 9 kolom, dengan enam fitur numerik, satu kolom nama mobil, satu kolom tahun model, dan satu target. Pemeriksaan menemukan 0 baris duplikat. Missing value ditemukan pada <i>Miles_per_Gallon</i> sebanyak 8 baris dan <i>Horsepower</i> sebanyak 6 baris; target tidak memiliki nilai kosong. Distribusi target adalah USA 254 (62,6%), Japan 79 (19,5%), dan Europe 73 (18,0%), sehingga kelas tidak seimbang. Tidak ditemukan nilai numerik nonpositif atau tanggal tahun yang gagal dikonversi. Pemeriksaan IQR menandai kandidat ekstrem pada MPG (1), Horsepower (8), dan Acceleration (6); kandidat tersebut tidak otomatis dihapus karena nilainya masih positif dan perlu ditinjau sebagai variasi mobil yang mungkin sah.", "Bodyx")

audit_rows=[[Paragraph("Pemeriksaan",styles["CellHead"]),Paragraph("Temuan",styles["CellHead"]),Paragraph("Keputusan",styles["CellHead"])],
 [Paragraph("Duplikat",styles["Cell"]),Paragraph("0 baris",styles["Cell"]),Paragraph("Tidak ada baris yang dihapus",styles["Cell"])],
 [Paragraph("Missing MPG",styles["Cell"]),Paragraph("8 nilai",styles["Cell"]),Paragraph("Median imputation dalam pipeline",styles["Cell"])],
 [Paragraph("Missing Horsepower",styles["Cell"]),Paragraph("6 nilai",styles["Cell"]),Paragraph("Median imputation dalam pipeline",styles["Cell"])],
 [Paragraph("Target",styles["Cell"]),Paragraph("USA 254; Japan 79; Europe 73",styles["Cell"]),Paragraph("Stratified split; Macro F1",styles["Cell"])]]
story.append(make_table(audit_rows,[30*mm,52*mm,82*mm]))
P("2. Feature Engineering dan Preprocessing", "H2x")
P("Kolom <i>Name</i> diekstraksi menjadi fitur <i>Brand</i> menggunakan token pertama nama mobil. Variasi ejaan/alias yang terlihat jelas pada data dinormalisasi: chevy/chevroelt menjadi chevrolet, maxda menjadi mazda, toyouta menjadi toyota, vokswagen/vw menjadi volkswagen, dan mercedes menjadi mercedes-benz. Kolom <i>Year</i> dikonversi dari tanggal menjadi <i>Model_Year</i>. Kolom nama penuh dan tanggal mentah tidak diteruskan ke model setelah fitur turunan dibuat.", "Bodyx")
P("Fitur numerik yang dipertahankan adalah Miles_per_Gallon, Cylinders, Displacement, Horsepower, Weight_in_lbs, Acceleration, dan Model_Year. Missing value numerik diimputasi dengan median dan ditambahkan indikator missing. Fitur Brand diimputasi dengan modus dan di-One-Hot Encode. Fitur numerik distandardisasi untuk Logistic Regression. Transformasi statistik tersebut berada di dalam Pipeline/ColumnTransformer sehingga hanya dipelajari dari training fold.", "Bodyx")
P("Split 80:20 dengan stratifikasi dilakukan sebelum imputasi, scaling, dan encoding: 324 baris training dan 82 baris test. Seleksi model menggunakan StratifiedKFold 5-fold dan Macro F1 hanya pada data training.", "Bodyx")
P("3. Perbandingan Model", "H2x")
cv_data=[[Paragraph("Model",styles["CellHead"]),Paragraph("Brand",styles["CellHead"]),Paragraph("CV Macro F1",styles["CellHead"]),Paragraph("Std",styles["CellHead"])]]
for _,r in CV.iterrows():
    cv_data.append([Paragraph(str(r['model']),styles["Cell"]),Paragraph("Ya" if bool(r['brand_feature']) else "Tidak",styles["Cell"]),Paragraph(f"{r['cv_f1_macro_mean']:.3f}",styles["Cell"]),Paragraph(f"{r['cv_f1_macro_std']:.3f}",styles["Cell"])])
story.append(make_table(cv_data,[59*mm,21*mm,38*mm,26*mm]))
P("Model terpilih adalah Logistic Regression dengan fitur Brand, menggunakan parameter terbaik C=10 dan tanpa class weight. Perbandingan tanpa Brand menunjukkan skor Macro F1 lebih rendah. Baseline DummyClassifier (prior) memperoleh CV Macro F1 0,257.", "Bodyx")
story.append(PageBreak())

P("HASIL EVALUASI DAN INTERPRETASI", "MainTitle")
P("UTS Intelegensi Buatan - Kasus 6 · NIM 24552011132", "SubTitle2")
P("4. Evaluasi Final pada Test Set", "H2x")
P("Setelah pemilihan model melalui cross-validation training set, model terpilih dievaluasi satu kali pada test set yang tidak digunakan dalam pelatihan maupun pemilihan model. Hasil pada 82 sampel test adalah sebagai berikut:", "Bodyx")
metrics=META["test_metrics"]
metric_rows=[[Paragraph("Metrik",styles["CellHead"]),Paragraph("Nilai",styles["CellHead"]),Paragraph("Interpretasi singkat",styles["CellHead"])],
 [Paragraph("Accuracy",styles["Cell"]),Paragraph(f"{metrics['accuracy']:.3f}",styles["Cell"]),Paragraph("Proporsi seluruh prediksi yang benar",styles["Cell"])],
 [Paragraph("Balanced Accuracy",styles["Cell"]),Paragraph(f"{metrics['balanced_accuracy']:.3f}",styles["Cell"]),Paragraph("Rata-rata recall antar kelas",styles["Cell"])],
 [Paragraph("Precision Macro",styles["Cell"]),Paragraph(f"{metrics['precision_macro']:.3f}",styles["Cell"]),Paragraph("Rata-rata precision ketiga kelas",styles["Cell"])],
 [Paragraph("Recall Macro",styles["Cell"]),Paragraph(f"{metrics['recall_macro']:.3f}",styles["Cell"]),Paragraph("Rata-rata recall ketiga kelas",styles["Cell"])],
 [Paragraph("Macro F1",styles["Cell"]),Paragraph(f"{metrics['f1_macro']:.3f}",styles["Cell"]),Paragraph("Keseimbangan precision dan recall per kelas",styles["Cell"])],
 [Paragraph("Weighted F1",styles["Cell"]),Paragraph(f"{metrics['f1_weighted']:.3f}",styles["Cell"]),Paragraph("F1 dirata-ratakan dengan bobot jumlah sampel",styles["Cell"])],
 [Paragraph("ROC-AUC OvR Macro",styles["Cell"]),Paragraph(f"{metrics['roc_auc_ovr_macro']:.3f}",styles["Cell"]),Paragraph("Kemampuan pemeringkatan antar kelas dengan skema one-vs-rest",styles["Cell"])]]
story.append(make_table(metric_rows,[40*mm,24*mm,104*mm]))
P("Confusion matrix menghasilkan 51/51 prediksi USA benar, 15/15 Europe benar, dan 16/16 Japan benar. Dengan demikian, semua metrik pada split test ini bernilai 1,000.", "Bodyx")
P("5. Interpretasi Kritis Fitur Brand dan Generalisasi", "H2x")
P("Hasil sempurna tidak boleh ditafsirkan sebagai bukti bahwa model pasti sempurna pada semua mobil di luar dataset. Setelah normalisasi terdapat 31 merek pada dataset penuh. Pemeriksaan hubungan Brand-Origin untuk interpretasi model dilakukan hanya pada training split; di training split ini setiap merek yang diamati berasosiasi dengan satu label Origin. Artinya, Brand menjadi proxy yang sangat kuat untuk target. Karena merek tersedia pada saat prediksi dilakukan, fitur ini dapat digunakan untuk tugas dalam ruang lingkup data ini, tetapi model dapat kesulitan pada merek yang belum pernah dilihat.", "Bodyx")
G=META["unseen_brand_group_cv_diagnostic"]
P(f"Sebagai pemeriksaan tambahan, dilakukan StratifiedGroupKFold 5-fold pada training set dengan grup berdasarkan Brand. Macro F1 rata-rata pada skenario merek yang ditahan dari fold pelatihan adalah {G['macro_f1_mean']:.3f} ± {G['macro_f1_std']:.3f}, lebih rendah daripada random stratified CV. Diagnostik ini menunjukkan bahwa performa random split sangat dipengaruhi kemunculan merek yang sama pada data training dan validasi. Pengujian grouped ini merupakan analisis generalisasi tambahan, bukan evaluasi test final kedua.", "Bodyx")
P("6. Kesimpulan", "H2x")
P("Pipeline Logistic Regression dengan fitur Brand merupakan model terbaik berdasarkan Macro F1 cross-validation pada data training dan memperoleh metrik sempurna pada test split yang ditetapkan. Proyek telah menerapkan split sebelum preprocessing, pipeline untuk imputasi/encoding/scaling, perbandingan dua algoritma, evaluasi multikelas, dan penyimpanan pipeline Joblib. Batasan utamanya adalah ukuran dataset yang relatif kecil, rentang tahun 1970-1982, dan hubungan satu-ke-satu antara Brand dan Origin pada data ini. Oleh sebab itu, klaim performa harus dibatasi pada distribusi data yang diuji; untuk merek baru, diagnostik grouped CV lebih relevan.", "Bodyx")
P("Artefak: fitur.py, latih.py, models/car_origin_pipeline.joblib, app.py, notebook/kasus06_colab.ipynb, dan output evaluasi. Aplikasi Streamlit menampilkan prediksi, probabilitas kelas, serta metrik evaluasi.", "Smallx")

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1")); canvas.line(18*mm, 14*mm, 192*mm, 14*mm)
    canvas.setFont("Helvetica",7); canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18*mm, 9*mm, "UTS Intelegensi Buatan - Kasus 6")
    canvas.drawRightString(192*mm, 9*mm, f"Halaman {doc.page}")
    canvas.restoreState()

OUT.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=14*mm,bottomMargin=20*mm,title="Laporan UTS AI Kasus 6")
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
