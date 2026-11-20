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
| 3 | *(TODO: training model)* | `models/model.pkl` |
| 4 | `streamlit run app.py` | Aplikasi web terbuka di browser |

`data_siap.csv` dan `model.pkl` sudah ada di repository, jadi aplikasi bisa langsung dijalankan dengan langkah 4 tanpa mengulang langkah 1–3.