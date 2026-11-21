# UTS Intelegensi Buatan - Kasus 6: Car Origin Classification

**NIM:** 24552011132  
**Kasus:** 6 (jumlah digit 26; 26 mod 10 = 6)  
**Dataset:** `kasus06_cars.csv`  
**Target:** `Origin` (`USA`, `Europe`, `Japan`)

## Ringkasan proyek

Proyek ini membangun sistem klasifikasi multikelas untuk memperkirakan `Origin` mobil berdasarkan informasi nama/merek, konsumsi bahan bakar, spesifikasi mesin, bobot, akselerasi, dan tahun model. Semua hasil angka di bawah dihasilkan dari eksekusi skrip `latih.py` pada dataset yang disertakan.

## Struktur folder

```text
UTS_AI_Kasus06/
├── dataset/
│   └── kasus06_cars.csv
├── analisis_eda.py
├── fitur.py
├── latih.py
├── app.py
├── requirements.txt
├── models/
│   ├── car_origin_pipeline.joblib
│   └── model_metadata.json
├── output/
│   ├── audit_data.json
│   ├── brand_origin_cross_tab.csv
│   ├── perbandingan_cv.csv
│   ├── metrics.json
│   ├── confusion_matrix.csv
│   ├── classification_report.csv
│   └── figures/
├── notebook/
│   └── kasus06_colab.ipynb
├── laporan/
│   └── laporan_uts_ai_kasus06.pdf
└── tests/
    └── test_pipeline.py
```

## Menjalankan secara lokal

Gunakan Python 3.10 atau lebih baru. Dari folder proyek:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python latih.py
streamlit run app.py
```

Jalankan `python analisis_eda.py` untuk membuat ringkasan EDA dan grafik, kemudian `python latih.py` untuk pemilihan model, evaluasi, dan penyimpanan pipeline. Skrip training membaca CSV dari `dataset/kasus06_cars.csv`, menyimpan pipeline ke `models/`, dan hasil analisis ke `output/`.

## Menjalankan di Google Colab

Buka `notebook/kasus06_colab.ipynb` melalui Google Colab. Notebook akan membuat folder `/content/dataset`. **File CSV akan berada di `/content/dataset/kasus06_cars.csv` dan terlihat melalui panel Files di sisi kiri Colab.**

Cara memasukkan CSV:

1. Jalankan sel persiapan folder.
2. Pada panel Files, refresh dan buka folder `dataset`.
3. Upload `kasus06_cars.csv` ke folder tersebut. Alternatifnya, jalankan sel upload manual di notebook; file akan dipindahkan ke folder `dataset`.
4. Jalankan sel berikutnya berurutan.

Notebook menulis ulang `fitur.py`, `latih.py`, dan `app.py` di runtime Colab sehingga semua berkas saling terhubung. Untuk membuka Streamlit dari Colab, notebook menggunakan tunnel sementara Cloudflare Quick Tunnel. URL tunnel bersifat sementara dan hanya aktif selama runtime berjalan.

## Metodologi

1. Audit dataset mentah: struktur, tipe data, missing value, duplikat, rentang numerik, distribusi target, dan variasi nama merek.
2. Feature engineering deterministik: ekstraksi `Brand` dari token pertama `Name`, normalisasi alias ejaan yang ditemukan pada dataset, serta ekstraksi `Model_Year` dari `Year`.
3. Train-test split 80:20 dengan `stratify=y` sebelum imputasi, encoding, dan scaling.
4. Preprocessing dilakukan di `Pipeline` dan `ColumnTransformer`. Imputasi median, indikator missing numerik, scaling numerik, imputasi modus dan One-Hot Encoding Brand dipelajari hanya dari training fold.
5. Pemilihan model memakai 5-fold `StratifiedKFold` dan Macro F1 pada training set. Dibandingkan Logistic Regression dan Random Forest; keduanya diuji dengan dan tanpa Brand.
6. Test set dievaluasi satu kali setelah model ditentukan.
7. Diagnostik tambahan `StratifiedGroupKFold` pada training set mengukur generalisasi saat suatu merek tidak muncul pada fold pelatihan. Diagnostik ini tidak menggantikan prosedur validasi utama.

## Ringkasan hasil eksperimen (dataset yang disertakan)

- Data: 406 baris, 9 kolom; 0 baris duplikat.
- Missing value: `Miles_per_Gallon` 8; `Horsepower` 6.
- Distribusi target: USA 254, Japan 79, Europe 73.
- Merek setelah normalisasi: 31 pada keseluruhan dataset. Hubungan Brand-Origin yang dipakai untuk interpretasi model dihitung hanya pada data training; di training split, setiap merek yang diamati berasosiasi dengan satu label Origin.
- Eksperimen CV terbaik: Logistic Regression + Brand, Macro F1 = 1.000 (5-fold stratified CV pada training set).
- Evaluasi final: 82 baris test; seluruh metrik utama tercatat 1.000 pada split ini.
- Diagnostik unseen-brand: Group CV Macro F1 = 0.602 ± 0.053. Hasil ini menunjukkan skor sempurna random split sangat bergantung pada merek yang sudah dikenal. Brand merupakan fitur yang tersedia saat prediksi (bukan kolom target atau informasi pascakejadian), tetapi berperan sebagai proxy kuat bagi Origin dalam dataset ini.

Angka final dan rincian lengkap ada pada `output/metrics.json` dan laporan PDF. Jangan mengubah angka laporan tanpa menjalankan ulang eksperimen.

## Keterbatasan dan interpretasi

- Dataset relatif kecil dan merepresentasikan mobil era 1970-1982; hasil tidak otomatis berlaku pada merek, negara, atau periode kendaraan lain.
- Pada data ini tiap merek hasil normalisasi hanya muncul dalam satu kategori Origin. Ini menjelaskan mengapa penggunaan Brand menghasilkan pemisahan kelas yang sangat kuat ketika merek yang sama ada di train dan test.
- Untuk mobil dari merek yang belum pernah dilihat model, diagnostik grouped CV lebih realistis daripada random split. Skornya lebih rendah, sehingga model tidak boleh diklaim telah tervalidasi sempurna untuk merek baru.
- Probabilitas yang ditampilkan Streamlit adalah probabilitas model, bukan jaminan asal manufaktur secara faktual.

## File output

- `models/car_origin_pipeline.joblib`: pipeline lengkap dan metadata.
- `models/model_metadata.json`: metadata model terpisah untuk inspeksi.
- `output/perbandingan_cv.csv`: perbandingan CV, termasuk ablation Brand.
- `output/metrics.json`: metadata dan metrik final.
- `output/confusion_matrix.csv`, `output/classification_report.csv`, serta grafik di `output/figures/`.

## Catatan kompatibilitas

Joblib/pickle sebaiknya dimuat pada lingkungan dengan versi Python dan scikit-learn yang kompatibel dengan versi training. Versi yang digunakan tercatat dalam `output/metrics.json`.
