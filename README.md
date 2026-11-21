# UTS Intelegensi Buatan – Kasus 6: Car Origin Classification

## Identitas Mahasiswa

| Keterangan | Informasi |
|---|---|
| Nama | Adika Pamungkas |
| NIM | 24552011132 |
| Kelas | TIF RP 23D |
| Mata Kuliah | Intelegensi Buatan |
| Dosen Pengampu | Muhammad Ikhwan Fathulloh, S.Kom. |
| Jenis Penilaian | Ujian Tengah Semester (UTS) |
| Kasus | Kasus 6 – Car Origin Classification |

---

## 1. Ringkasan Proyek

Proyek ini merupakan implementasi tugas Ujian Tengah Semester (UTS) mata kuliah Intelegensi Buatan dengan topik **Car Origin Classification**.

Tujuan proyek adalah membangun sistem klasifikasi multikelas untuk memprediksi negara asal mobil (*Origin*) berdasarkan sejumlah karakteristik kendaraan, seperti nama dan merek mobil, konsumsi bahan bakar, spesifikasi mesin, bobot, akselerasi, serta tahun model.

Sistem dikembangkan menggunakan Python dengan pendekatan Machine Learning. Model yang dibangun digunakan untuk mengklasifikasikan mobil ke dalam tiga kategori:

- USA
- Europe
- Japan

Selain proses pengembangan dan evaluasi model, proyek ini juga dilengkapi dengan analisis eksplorasi data (EDA), feature engineering, visualisasi, notebook Google Colab, pengujian pipeline, serta aplikasi interaktif berbasis Streamlit.

## 2. Informasi Dataset

| Keterangan | Informasi |
|---|---|
| Nama Dataset | `kasus06_cars.csv` |
| Jumlah Data | 406 baris |
| Jumlah Kolom | 9 kolom |
| Target | `Origin` |
| Kategori Target | USA, Europe, Japan |
| Missing Value | Miles_per_Gallon: 8, Horsepower: 6 |
| Duplikasi | 0 baris |

Dataset digunakan sebagai sumber utama dalam proses analisis, preprocessing, pelatihan, dan evaluasi model.

## 3. Struktur Folder Proyek

```text
UTS_Artificial_Intelligence_Adika_Pamungkas_24552011132/
|
|-- archive/
|   `-- create_report.py
|
|-- dataset/
|   `-- kasus06_cars.csv
|
|-- laporan/
|   `-- laporan_uts_ai_kasus06.pdf
|
|-- models/
|   |-- car_origin_pipeline.joblib
|   `-- model_metadata.json
|
|-- notebook/
|   `-- kasus06_colab.ipynb
|
|-- output/
|   |-- audit_data.json
|   |-- brand_frequency.csv
|   |-- brand_origin_cross_tab.csv
|   |-- classification_report.csv
|   |-- confusion_matrix.csv
|   |-- data_dictionary.csv
|   |-- figures/
|   |   |-- confusion_matrix.png
|   |   |-- correlation_heatmap.png
|   |   |-- numeric_distributions.png
|   |   |-- perbandingan_cv.png
|   |   `-- target_distribution.png
|   |-- metrics.json
|   |-- perbandingan_cv.csv
|   `-- training_log.txt
|
|-- tests/
|   `-- test_pipeline.py
|
|-- analisis_eda.py
|-- app.py
|-- fitur.py
|-- latih.py
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## 4. Teknologi yang Digunakan

- **Python** – bahasa pemrograman utama.
- **Pandas dan NumPy** – pengolahan dan manipulasi data.
- **Scikit-learn** – preprocessing, pemilihan model, pelatihan, dan evaluasi.
- **Matplotlib dan Seaborn** – visualisasi data.
- **Joblib** – penyimpanan dan pemuatan model.
- **Streamlit** – pengembangan aplikasi prediksi interaktif.
- **Google Colab** – lingkungan pengembangan dan eksekusi notebook.
- **Git dan GitHub** – pengelolaan versi dan dokumentasi proyek.

## 5. Menjalankan Proyek Secara Lokal

Pastikan Python versi 3.10 atau lebih baru sudah terinstal.

### A. Clone Repository

```bash
git clone https://github.com/Alpmngks310/UTS_Artificial_Intelligence_Adika_Pamungkas_24552011132.git
```

Masuk ke folder proyek:

```bash
cd UTS_Artificial_Intelligence_Adika_Pamungkas_24552011132
```

### B. Membuat Virtual Environment

```bash
python -m venv .venv
```

Aktifkan virtual environment di Windows:

```bash
.venv\Scripts\activate
```

### C. Instalasi Dependencies

```bash
pip install -r requirements.txt
```

### D. Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan berjalan melalui browser dan menyediakan antarmuka untuk melakukan prediksi asal mobil berdasarkan karakteristik kendaraan.

**Catatan:** Model hasil pelatihan sudah tersedia di folder `models/`, sehingga untuk menjalankan aplikasi tidak perlu melatih ulang model terlebih dahulu.

Jika ingin mengulang proses analisis dan pelatihan dari awal, jalankan:

```bash
python analisis_eda.py
python latih.py
```

Skrip akan membaca dataset dari folder `dataset/`, kemudian menyimpan hasil pelatihan dan evaluasi ke folder `models/` dan `output/`.

## 6. Menjalankan Proyek di Google Colab

Proyek juga menyediakan notebook:

`notebook/kasus06_colab.ipynb`

Notebook tersebut dapat dibuka melalui Google Colab untuk menjalankan proses analisis dan pelatihan model.

Langkah penggunaan:

1. Buka file notebook melalui Google Colab.
2. Jalankan sel persiapan folder.
3. Pastikan dataset `kasus06_cars.csv` tersedia di `/content/dataset/`.
4. Jika belum tersedia, upload dataset melalui panel Files atau gunakan sel upload manual.
5. Jalankan sel notebook secara berurutan.

Notebook akan menyiapkan dan menulis file `fitur.py`, `latih.py`, serta `app.py` pada runtime Colab agar komponen proyek saling terhubung.

Untuk menjalankan aplikasi Streamlit dari Colab, notebook menggunakan Cloudflare Quick Tunnel. URL yang dihasilkan bersifat sementara dan hanya aktif selama runtime berjalan.

## 7. Metodologi dan Tahapan Pengembangan

Proses pengembangan model dilakukan melalui beberapa tahapan berikut.

1. **Exploratory Data Analysis (EDA)**  
   Melakukan pemeriksaan struktur dataset, tipe data, missing value, duplikasi, rentang nilai numerik, distribusi target, dan variasi nama merek.

2. **Feature Engineering**  
   Mengekstraksi fitur `Brand` dari token pertama kolom `Name`, melakukan normalisasi alias ejaan merek, serta mengekstraksi `Model_Year` dari kolom `Year`.

3. **Train-Test Split**  
   Membagi dataset menjadi data training dan testing dengan proporsi 80:20 menggunakan stratifikasi berdasarkan target `Origin`.

4. **Data Preprocessing**  
   Menggunakan `Pipeline` dan `ColumnTransformer` untuk menangani missing value, scaling fitur numerik, serta encoding fitur kategorikal. Seluruh preprocessing dipelajari dari training fold untuk menghindari data leakage.

5. **Model Selection**  
   Membandingkan Logistic Regression dan Random Forest dengan variasi penggunaan fitur `Brand` dan tanpa fitur `Brand`. Evaluasi menggunakan 5-fold Stratified Cross-Validation dengan Macro F1 sebagai metrik utama.

6. **Final Evaluation**  
   Melakukan evaluasi pada test set setelah proses pemilihan model selesai.

7. **Generalization Diagnostic**  
   Menggunakan StratifiedGroupKFold untuk mengukur kemampuan model dalam memprediksi data dengan merek yang tidak muncul pada fold pelatihan.

## 8. Hasil Eksperimen

Berdasarkan hasil eksperimen pada dataset yang disertakan, diperoleh ringkasan berikut.

| Komponen | Hasil |
|---|---|
| Model Terbaik | Logistic Regression + Brand |
| Stratified CV Macro F1 | 1.000 |
| Jumlah Data Test | 82 |
| Metrik Utama pada Test Set | 1.000 |
| Group CV Macro F1 | 0.602 ± 0.053 |
| Jumlah Merek Setelah Normalisasi | 31 |

Hasil tersebut menunjukkan bahwa Logistic Regression dengan fitur Brand memberikan performa sangat tinggi pada skenario random stratified cross-validation dan test split.

Namun, hasil Group CV menunjukkan bahwa kemampuan generalisasi terhadap merek yang belum pernah dilihat model lebih rendah.

Fitur Brand merupakan proxy yang sangat kuat untuk kelas Origin pada dataset ini. Oleh karena itu, skor sempurna pada random split tidak boleh diartikan sebagai jaminan bahwa model akan memiliki performa sempurna pada merek kendaraan baru.

Rincian hasil eksperimen dapat ditemukan pada:

- `output/metrics.json`
- `output/perbandingan_cv.csv`
- `output/classification_report.csv`
- `output/confusion_matrix.csv`
- `output/figures/`

## 9. Keterbatasan Model

Beberapa keterbatasan yang perlu diperhatikan:

- Dataset relatif kecil, yaitu 406 baris.
- Dataset merepresentasikan kendaraan pada era 1970–1982 sehingga hasilnya tidak otomatis berlaku pada kendaraan modern.
- Setiap merek setelah normalisasi pada dataset ini hanya muncul dalam satu kategori Origin.
- Penggunaan fitur Brand memberikan pengaruh besar terhadap hasil klasifikasi.
- Kemampuan generalisasi terhadap merek yang belum pernah ditemukan model masih terbatas.
- Probabilitas yang ditampilkan aplikasi merupakan probabilitas prediksi model, bukan jaminan faktual mengenai negara asal kendaraan.

## 10. File Penting dalam Proyek

| File | Fungsi |
|---|---|
| `analisis_eda.py` | Analisis eksplorasi dataset |
| `fitur.py` | Feature engineering |
| `latih.py` | Pelatihan dan evaluasi model |
| `app.py` | Aplikasi prediksi berbasis Streamlit |
| `car_origin_pipeline.joblib` | Pipeline model yang sudah dilatih |
| `model_metadata.json` | Metadata model |
| `kasus06_colab.ipynb` | Notebook Google Colab |
| `test_pipeline.py` | Pengujian pipeline |
| `metrics.json` | Ringkasan metrik eksperimen |
| `training_log.txt` | Catatan proses pelatihan |

## 11. Penutup

Proyek UTS Intelegensi Buatan Kasus 6 ini mengimplementasikan proses klasifikasi multikelas menggunakan pendekatan Machine Learning, mulai dari analisis data, preprocessing, feature engineering, pemilihan model, evaluasi, hingga implementasi aplikasi prediksi interaktif.

Selain menghasilkan model dengan performa tinggi pada skenario evaluasi utama, proyek ini juga memperhatikan aspek validasi dan keterbatasan generalisasi model, khususnya ketika menghadapi merek kendaraan yang belum pernah ditemukan selama proses pelatihan.

Dengan demikian, proyek ini tidak hanya berfokus pada pencapaian nilai evaluasi, tetapi juga pada pemahaman proses pengembangan model Machine Learning dan interpretasi hasilnya secara lebih kritis.

---

**Repository:** [UTS Artificial Intelligence – Adika Pamungkas](https://github.com/Alpmngks310/UTS_Artificial_Intelligence_Adika_Pamungkas_24552011132)

**Nama:** Adika Pamungkas  
**NIM:** 24552011132  
**Kelas:** TIF RP 23D  
**Mata Kuliah:** Intelegensi Buatan
