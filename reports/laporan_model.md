# Laporan Bagian Model — Prediksi Kualitas Udara Jakarta (ISPU)

Dokumen ini berisi draf bagian **Metode Pemodelan, Hasil Eksperimen, Perbandingan Model, Analisis Kesalahan, dan Pertimbangan Etika (LO 6)** yang dapat langsung digunakan untuk laporan dan presentasi tim. Seluruh angka dan grafik dihasilkan dari eksekusi bersih `src/train_model.ipynb` dengan `scikit-learn==1.9.1`.

---

## 1. Metode Pemodelan

### 1.1 Fitur, Target, dan Kontrak Integrasi
Model menggunakan tepat **4 fitur** dan **1 target** sesuai kontrak di `src/config.py`:
- **Fitur (`FITUR`):** `ispu_hari_ini`, `ispu_kemarin`, `bulan` (1–12), dan `hari` (0 = Senin … 6 = Minggu).
- **Target (`TARGET`):** `ispu_besok` (angka ISPU 1 hari sesudahnya).
- Kolom identitas (`tanggal`, `stasiun`) **tidak dimasukkan** sebagai fitur.
- Model mengeluarkan **angka ISPU**, sedangkan konversi ke label kategori (`BAIK`, `SEDANG`, `TIDAK SEHAT`, `SANGAT TIDAK SEHAT`, `BERBAHAYA`) dilakukan oleh aplikasi memakai `kategori_ispu()`.

### 1.2 Pembagian Data Kronologis (Tanpa Pengacakan)
Karena data berupa deret waktu harian (22.835 baris, 2 Jan 2010 – 27 Feb 2025, sudah diurutkan per tanggal), pembagian dilakukan secara **kronologis** untuk mencegah *data leakage* dari masa depan:

| Bagian Data | Jumlah Baris | Rentang Tanggal | Baik | Sedang | Tidak Sehat+ (>100) | Peran |
|---|---:|---|---:|---:|---:|---|
| **Sub-latih (64%)** | 14.614 | 2010-01-02 s.d. 2019-06-05 | 20,7% | 65,1% | 14,2% | Melatih model saat eksperimen & *tuning* |
| **Validasi (16%)** | 3.654 | 2019-06-06 s.d. 2021-08-01 | 13,9% | 70,8% | 15,3% | Memilih model & setelan terbaik |
| **Total Latih (80%)** | 18.268 | 2010-01-02 s.d. 2021-08-01 | 19,4% | 66,3% | 14,4% | Melatih ulang model final |
| **Uji (20%)** | 4.567 | 2021-08-02 s.d. 2025-02-27 | 12,4% | 77,4% | 10,2% | Evaluasi akhir (**dipakai sekali saja**) |

### 1.3 Metrik Evaluasi & Baseline
Sebagai pembanding minimal, digunakan **Baseline Persistensi** (`ISPU besok = ISPU hari ini`). Selain kesalahan numerik (**MAE** dan **RMSE**), kinerja operasional diukur dengan:
1. **Akurasi Kategori (%):** persentase kesesuaian kategori hasil konversi `kategori_ispu()` antara prediksi dan aktual.
2. **TS Ketangkep / Recall > 100 (%):** persentase hari yang aslinya Tidak Sehat ke atas (ISPU > 100) yang berhasil diprediksi > 100. Metrik ini krusial bagi kesehatan publik.
3. **Alarm Benar / Precision > 100 (%):** persentase prediksi > 100 yang benar-benar terjadi > 100 (mengukur tingkat alarm palsu).
4. **F1 Tidak Sehat (%):** rata-rata harmonik antara *TS Ketangkep* dan *Alarm Benar*.

**Kriteria pemilihan model:** model terbaik harus **mengalahkan baseline pada keempat metrik utama sekaligus** (MAE lebih rendah, Akurasi Kategori lebih tinggi, TS Ketangkep lebih tinggi, dan Alarm Benar lebih tinggi) pada tahap validasi, serta berukuran jauh di bawah batas 100 MB GitHub.

### 1.4 Keluarga Model & Penanganan Ketidakseimbangan Data
Empat keluarga model regresi dibandingkan:
1. **Linear Regression** — model linear dasar.
2. **Ridge + One-Hot Encoding `bulan`** — dibungkus dalam `sklearn.pipeline.Pipeline` (input tetap 4 kolom `FITUR`) agar dapat menangkap pola musiman non-linear dari EDA (ISPU naik hingga puncak Oktober lalu turun tajam di Desember).
3. **Random Forest Regressor** — ansambel pohon non-linear dengan pembatasan kedalaman & daun (`max_depth=10`, `min_samples_leaf=20/25`) untuk mencegah *overfitting* sekaligus menjaga ukuran file jauh di bawah 100 MB (setelan bawaan Random Forest mencapai ~159 MB).
4. **Gradient Boosting Regressor (`loss="huber"`)** — model *boosting* dengan *Huber loss* agar tahan terhadap lonjakan ekstrem ISPU.

Untuk mengatasi dominasi kelas mayoritas `SEDANG` (66,3% di data latih) yang membuat regresi biasa "main aman ke tengah", dibandingkan dua pendekatan **hanya pada data latih** (data validasi dan uji tetap asli):
- **Undersampling (`Sedang dipotong`):** memotong jumlah hari `SEDANG` sebanyak jumlah hari `BAIK` secara acak.
- **Pembobotan deterministik (`sample_weight`):** menurunkan bobot observasi `SEDANG` menjadi `w_sedang < 1,0` sementara kelas lain berbobot `1,0`. Pendekatan ini mempertahankan 100% data latih dan 100% reprodusibel tanpa variasi acak.

---

## 2. Hasil Eksperimen & Pemilihan Model (Data Validasi)

Seluruh penyetelan dilakukan pada `sub_latih` (2010–2019) → `validasi` (2019–2021):

| Model | Setelan | MAE | RMSE | Akurasi Kategori | TS Ketangkep | Alarm Benar | F1 TS | Kalahkan Semua Baseline |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| Baseline (besok = hari ini) | `ispu_besok = ispu_hari_ini` | 16,02 | 22,95 | 72,6% | 49,1% | 48,7% | 48,9% | — |
| Linear Regression | default | 14,54 | 20,40 | 75,0% | 37,3% | 55,9% | 44,8% | Tidak |
| Ridge + OHE bulan | `alpha=10`, OHE bulan | 14,37 | 20,22 | 75,7% | 38,2% | 55,7% | 45,3% | Tidak |
| Random Forest | 200 pohon, `d=10`, `leaf=20` | 14,49 | 20,23 | 75,4% | 30,0% | 57,9% | 39,5% | Tidak |
| Gradient Boosting Huber | `huber`, `d=3`, `leaf=30`, `lr=0.05`, `n=150` | **14,25** | 20,26 | **76,2%** | 21,6% | **68,4%** | 32,8% | Tidak |
| Linear Regression (Sedang dipotong) | undersample `SEDANG=BAIK` | 15,07 | 21,00 | 74,2% | 47,1% | 50,3% | 48,7% | Tidak |
| Ridge + OHE bulan (Sedang dipotong) | `alpha=10`, undersample `SEDANG=BAIK` | 15,11 | 20,94 | 74,2% | 48,9% | 49,2% | 49,1% | Tidak |
| Ridge + OHE bulan (`w_sedang=0.28`) | `alpha=10`, `sample_weight` 0,28 | 15,21 | 21,03 | 74,2% | 50,0% | 48,8% | 49,4% | **Ya** |
| **Ridge + OHE bulan (`w_sedang=0.30`)** | `alpha=10`, `sample_weight` 0,30 | 15,15 | 20,98 | **74,2%** | **49,3%** | **49,0%** | 49,2% | **Ya** |
| Ridge + OHE bulan (`w_sedang=0.35`) | `alpha=10`, `sample_weight` 0,35 | 15,02 | 20,85 | 74,4% | 48,2% | 49,5% | 48,8% | Tidak |
| Random Forest (`w_sedang=0.55`) | 200 pohon, `d=10`, `leaf=25`, `w=0.55` | 14,95 | 20,63 | 73,8% | 48,9% | 48,2% | 48,5% | Tidak |
| Random Forest (`w_sedang=0.60`) | 200 pohon, `d=10`, `leaf=25`, `w=0.60` | 14,83 | 20,53 | 74,2% | 46,2% | 49,4% | 47,8% | Tidak |
| Gradient Boosting Huber (`w_sedang=0.43`) | `huber`, `d=3`, `leaf=30`, `w=0.43` | 14,77 | 20,53 | 73,9% | 50,0% | 49,7% | 49,9% | **Ya** |
| **Gradient Boosting Huber (`w_sedang=0.45`)** | `huber`, `d=3`, `leaf=30`, `w=0.45` | 14,69 | 20,45 | 74,1% | 49,3% | 50,2% | 49,7% | **Ya** |
| Gradient Boosting Huber (`w_sedang=0.48`) | `huber`, `d=3`, `leaf=30`, `w=0.48` | 14,63 | 20,42 | 74,4% | 47,5% | 50,9% | 49,1% | Tidak |

**Temuan validasi:**
- Seluruh model tanpa penyeimbangan kalah jauh dari baseline dalam menangkap hari Tidak Sehat (21,6–38,2% vs baseline 49,1%).
- Dua kandidat utama lolos syarat mengalahkan baseline di semua metrik: **`Ridge + OHE bulan (w_sedang=0.30)`** (akurasi kategori tertinggi 74,2% dengan parameter sederhana & deterministik) dan **`Gradient Boosting Huber (w_sedang=0.45)`** (MAE terendah 14,69).

---

## 3. Hasil Evaluasi Final di Data Uji (Agu 2021 – Feb 2025, n = 4.567)

Model referensi dan kandidat terpilih dilatih ulang pada seluruh 80% data latih (18.268 baris) lalu dievaluasi **sekali** pada 20% data uji:

| Model | Setelan | MAE | RMSE | Akurasi Kategori | TS Ketangkep | Alarm Benar | F1 TS | Ukuran File | Kalahkan Semua Baseline |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Baseline (besok = hari ini) | `ispu_besok = ispu_hari_ini` | 11,42 | 16,05 | 79,4% | 46,9% | 46,8% | 46,8% | — | — |
| Linear Regression | default | 10,67 | 14,43 | 81,8% | 31,0% | 64,6% | 41,9% | 0,001 MB | Tidak |
| Ridge + OHE bulan | `alpha=10`, OHE bulan | **10,55** | **14,30** | **82,2%** | 31,4% | 63,8% | 42,1% | 0,003 MB | Tidak |
| Random Forest | 200 pohon, `d=10`, `leaf=20` | 10,79 | 14,55 | 81,2% | 20,4% | 57,6% | 30,2% | 7,601 MB | Tidak |
| Gradient Boosting Huber | `huber`, `d=3`, `leaf=30`, `lr=0.05`, `n=150` | 10,63 | 14,53 | 81,3% | 9,7% | **73,8%** | 17,1% | 0,200 MB | Tidak |
| Linear Regression (Sedang dipotong) | undersample `SEDANG=BAIK` | 11,01 | 14,90 | 81,5% | 48,8% | 52,5% | **50,6%** | 0,001 MB | **Ya** |
| **Ridge + OHE bulan (`w_sedang=0.30`) ★** | `alpha=10`, OHE bulan, `w_sedang=0.30` | **11,24** | **15,06** | **80,4%** | **52,3%** | **48,6%** | **50,4%** | **0,003 MB** | **Ya** |
| Random Forest (`w_sedang=0.55`) | 200 pohon, `d=10`, `leaf=25`, `w=0.55` | 11,17 | 14,97 | 79,6% | 50,8% | 46,6% | 48,6% | 6,934 MB | Tidak |
| Gradient Boosting Huber (`w_sedang=0.45`) | `huber`, `d=3`, `leaf=30`, `w=0.45` | 11,03 | 14,92 | 79,5% | 47,3% | 48,0% | 47,7% | 0,195 MB | **Ya** |

Grafik pendukung tersimpan di:
- `reports/figures/7_perbandingan_model.png` — perbandingan 4 metrik utama terhadap garis baseline.
- `reports/figures/8_confusion_matrix_model.png` — *confusion matrix* kategori Linear Regression standar vs model terpilih.
- `reports/figures/9_prediksi_vs_aktual_dan_musiman.png` — sebaran prediksi vs aktual serta efek musiman bulanan yang dipelajari model.

### Mengapa `Ridge + OHE bulan (w_sedang=0.30)` Dipilih sebagai Model Final?
1. **Konsisten mengalahkan baseline di seluruh 4 metrik baik pada validasi maupun uji.** Pada data uji, model menurunkan MAE (11,24 vs 11,42) dan RMSE (15,06 vs 16,05), menaikkan akurasi kategori (+1,0 poin ke 80,4%), menaikkan deteksi hari Tidak Sehat (+5,4 poin ke **52,3%** — naik +21,3 poin dari Linear Regression standar 31,0%), serta menaikkan ketepatan alarm (+1,8 poin ke 48,6%) dan F1 Tidak Sehat (+3,6 poin ke 50,4%).
2. **Terkalibrasi baik untuk hari Tidak Sehat.** Linear Regression standar hanya berani menebak > 100 di 4,9% hari uji (rata-rata prediksi 94,3 saat aktual 113,5), sedangkan model terpilih menebak > 100 di **10,9%** hari uji — sangat dekat dengan proporsi aktual **10,2%** — dengan rata-rata prediksi **101,5** pada hari Tidak Sehat.
3. **Menangkap pola musiman EDA secara langsung dan interpretabel.** Koefisien *One-Hot* `bulan` yang dipelajari model naik bertahap dari Januari (`+0,0`) hingga puncak September (`+11,8`) dan Oktober (`+11,9`), lalu turun tajam kembali di Desember (`+0,0`), sesuai persis dengan grafik musiman pada EDA. Koefisien fitur lain juga konsisten dengan EDA: `ispu_hari_ini` (`+0,619`) sebagai prediktor terkuat, diikuti `ispu_kemarin` (`+0,245`), dan pengaruh `hari` yang kecil (`-0,481`).
4. **Deterministik, lebih stabil dari model pohon, dan sangat ringan.** Berbeda dari *undersampling* acak yang membuang ~68% baris `SEDANG`, `sample_weight` memakai 100% data latih tanpa unsur acak. Dibanding `Gradient Boosting Huber (w=0.45)`, `Ridge + OHE (w=0.30)` lebih tahan terhadap pergeseran distribusi pasca-2021 (TS ketangkep 52,3% vs 47,3%, akurasi 80,4% vs 79,5%). Ukuran `models/model.pkl` hanya **3.292 bytes (0,003 MB)**, memuat instan di Streamlit Cloud, dan hanya memakai kelas standar `scikit-learn`.

---

## 4. Analisis Kesalahan, Kinerja per Stasiun, dan Keterbatasan

### 4.1 Kinerja per Stasiun (Data Uji)

| Stasiun | Jumlah Hari Uji | Hari TS Aktual | MAE | Akurasi Kategori | TS Ketangkep | Alarm Benar |
|---|---:|---:|---:|---:|---:|---:|
| DKI1 (Bunderan HI) | 903 | 32 | 10,17 | 79,5% | 18,8% | 14,6% |
| DKI2 (Kelapa Gading) | 924 | 62 | 10,08 | 82,3% | 29,0% | 24,7% |
| DKI3 (Jagakarsa) | 921 | 39 | 11,46 | 87,8% | 17,9% | 20,6% |
| DKI4 (Lubang Buaya) | 897 | 249 | 13,33 | 77,7% | 71,1% | 67,8% |
| DKI5 (Kebon Jeruk) | 922 | 83 | 11,21 | 74,6% | 42,2% | 38,5% |

- Lebih dari separuh hari Tidak Sehat pada periode uji (249 dari 465 hari, atau 53,5%) terjadi di **DKI4 (Lubang Buaya)**, dan model berhasil mendeteksi **71,1%** di antaranya dengan ketepatan alarm **67,8%**.
- Pada stasiun yang jarang mengalami hari Tidak Sehat (DKI1 & DKI3), sebagian besar kejadian Tidak Sehat berupa lonjakan singkat sehingga lebih sulit diprediksi hanya dari histori ISPU.

### 4.2 Keterbatasan Model & Catatan Target Rubrik "85%"
1. **Lonjakan mendadak (53,1%):** di data uji, **53,1%** hari Tidak Sehat terjadi setelah hari yang ISPU-nya masih ≤ 100. Tanpa sinyal eksternal (misalnya pola angin mikro, inversi suhu lokal, atau kejadian emisi mendadak), batas atas deteksi hari Tidak Sehat secara alamiah terbatas di sekitar ~50%.
2. **Upaya fitur tambahan yang sudah diuji tim:** penambahan fitur lag 2 hari / rata-rata 7 hari / identitas stasiun maupun data cuaca ERA5 Open-Meteo (hujan, angin, suhu) hanya menaikkan akurasi ±0,5 poin (MAE 10,67 → 10,39) sehingga tidak sepadan dengan kompleksitas integrasi API cuaca di aplikasi.
3. **Target rubrik "85% akurat":** pada seluruh eksperimen (termasuk model non-linear seperti Random Forest dan Gradient Boosting), akurasi kategori maksimal di data uji berada di **80,4%–82,2%** (meski pada stasiun tertentu seperti DKI3 mencapai 87,8%). Hal ini wajar untuk peramalan kualitas udara harian berfitur historis singkat dan menunjukkan perlunya transparansi keterbatasan model saat presentasi.

---

## 5. Pertimbangan Etika & Dampak Sosial (LO 6)

1. **Asimetri Risiko Kesalahan (*False Negative* vs *False Positive*):** Dalam konteks kesehatan publik, **kesalahan paling berbahaya adalah *false negative*** — model memprediksi "Sedang" padahal kualitas udara besok "Tidak Sehat", sehingga kelompok rentan (anak-anak, lansia, penderita asma) tetap beraktivitas di luar tanpa perlindungan. Oleh karena itu, model sengaja diimbangi (`w_sedang=0.30`) agar mendeteksi **52,3%** hari Tidak Sehat (naik dari 31,0% pada model standar), menerima sedikit tambahan alarm palsu demi keselamatan pengguna.
2. **Keterwakilan Spasial:** Data hanya berasal dari **5 stasiun pemantau resmi** di DKI Jakarta. Seperti terlihat pada tabel per stasiun, pola polusi berbeda jauh antarwilayah (DKI4 jauh lebih sering Tidak Sehat dibanding DKI1/DKI3). Hasil prediksi satu stasiun belum tentu mencerminkan kualitas udara mikro di seluruh lingkungan Jakarta.
3. **Transparansi kepada Pengguna:** Aplikasi Streamlit (Orang 3) perlu mencantumkan keterangan jelas (*disclaimer*) bahwa angka dan kategori yang ditampilkan adalah **hasil prediksi statistik berbasis data historis**, bukan pengumuman resmi pemerintah/BMKG/DLH DKI Jakarta, serta menampilkan ringkasan tingkat kesalahan model secara jujur.
