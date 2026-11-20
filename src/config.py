from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = ROOT / "data"/ "raw"
DATA_RAW_POLA = "ispu_dki[1-5].csv"
DATA_SIAP = ROOT / "data" / "processed" / "data_siap.csv"
RINGKASAN_DATA = ROOT / "data" / "processed" / "ringkasan_data.txt"
MODEL_PATH = ROOT / "models" / "model.pkl"
FOLDER_GRAFIK = ROOT / "reports" / "figures"



FITUR = ["ispu_hari_ini", "ispu_kemarin", "bulan", "hari"]
TARGET = "ispu_besok"
KOLOM_ID = ["tanggal", "stasiun"]

BATAS_KATEGORI = [
    (50, "BAIK"),
    (100, "SEDANG"),
    (200, "TIDAK SEHAT"),
    (300, "SANGAT TIDAK SEHAT"),
]
KATEGORI_TERTINGGI = "BERBAHAYA"


def kategori_ispu(nilai):
    for batas_atas, nama in BATAS_KATEGORI:
        if nilai <= batas_atas:
            return nama
    return KATEGORI_TERTINGGI

