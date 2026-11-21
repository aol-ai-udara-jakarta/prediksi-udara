# Prediksi Kualitas Udara Jakarta

Aplikasi web untuk memprediksi ISPU (Indeks Standar Pencemar Udara) besok di 5 stasiun pemantau DKI Jakarta.

## Setup

Butuh **Python 3.12** dan **Git**.

```
git clone https://github.com/aol-ai-udara-jakarta/prediksi-udara.git
cd prediksi-udara
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

- Di Mac/Linux, ganti baris aktivasi dengan `source venv/bin/activate`.
- Kalau muncul error *"running scripts is disabled"* di PowerShell, jalankan sekali:
  ```
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```
- Setiap membuka terminal baru, aktifkan lagi venv dengan `venv\Scripts\activate`. Pastikan muncul `(venv)` di depan baris terminal.
- Untuk membuka notebook (`.ipynb`) di VS Code, pilih **Select Kernel → Python Environments → venv**.

## Cara menjalankan

Jalankan berurutan. Untuk setiap notebook, klik **Restart → Run All**.

| Urutan | File | Hasil |
|---|---|---|
| 1 | `src/prepare_data.ipynb` | `data/processed/data_siap.csv` |
| 2 | `src/eda.ipynb` | Grafik di `reports/figures/` |
| 3 | `src/train_model.ipynb` | `models/model.pkl` & grafik evaluasi di `reports/figures/` |
| 4 | `streamlit run app.py` | Aplikasi web terbuka di browser |

`data_siap.csv` dan `model.pkl` sudah ada di repository, jadi aplikasi bisa langsung dijalankan dengan langkah 4 tanpa mengulang langkah 1–3.

## Model

- **File training:** `src/train_model.ipynb` (buka di VS Code dengan kernel `venv`, lalu jalankan **Restart → Run All**).
- **Output:** `models/model.pkl` (±3 KB, jauh di bawah batas 100 MB GitHub) dan 3 grafik evaluasi (`7_` s.d. `9_`) di `reports/figures/`.
- **Model terpilih:** `Ridge + One-Hot Encoding (bulan)` dengan pembobotan kelas `SEDANG = 0.30` pada data latih saja (dipilih melalui validasi kronologis karena mengalahkan baseline pada seluruh 4 metrik evaluasi: MAE 11,24, akurasi kategori 80,4%, deteksi hari Tidak Sehat 52,3%, dan alarm benar 48,6% pada data uji Agu 2021 – Feb 2025).
- **Kontrak integrasi (untuk `app.py`):** muat dengan `joblib.load(MODEL_PATH)`, kirim `DataFrame` berkolom `FITUR` (`["ispu_hari_ini", "ispu_kemarin", "bulan", "hari"]`) dalam urutan persis seperti di `src/config.py`. Output berupa angka prediksi ISPU besok; konversi ke kategori dilakukan aplikasi dengan `kategori_ispu()`.
- **Draf laporan bagian model:** `reports/laporan_model.md`.
