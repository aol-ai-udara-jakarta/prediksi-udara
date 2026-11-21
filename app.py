import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import (DATA_RAW_DIR, DATA_RAW_POLA, FITUR, MODEL_PATH,
                    kategori_ispu)

ISPU_MAKS_WAJAR = 500
BATAS_TIDAK_SEHAT = 100
JUMLAH_HARI_GRAFIK = 6

NAMA_BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
              "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

NAMA_POLUTAN = {"PM25": "PM2.5", "PM10": "PM10", "SO2": "SO₂",
                "CO": "CO", "O3": "O₃", "NO2": "NO₂"}

GAYA_KATEGORI = {
    "BAIK": {"teks": "#1b5e20", "latar": "#e8f5e9"},
    "SEDANG": {"teks": "#0d47a1", "latar": "#e3f2fd"},
    "TIDAK SEHAT": {"teks": "#8a5a00", "latar": "#fff6d6"},
    "SANGAT TIDAK SEHAT": {"teks": "#b71c1c", "latar": "#fdecea"},
    "BERBAHAYA": {"teks": "#ffffff", "latar": "#212121"},
}


def saran(kategori):
    if kategori == "BAIK":
        return "udara bersih, aman untuk olahraga dan aktivitas di luar ruangan."
    elif kategori == "SEDANG":
        return ("masih aman untuk beraktivitas di luar. Kelompok sensitif "
                "(anak-anak, lansia, penderita asma) sebaiknya mengurangi "
                "aktivitas berat yang lama di luar ruangan.")
    elif kategori == "TIDAK SEHAT":
        return ("pakai masker saat di luar, kurangi olahraga outdoor, dan "
                "tutup jendela di siang hari.")
    elif kategori == "SANGAT TIDAK SEHAT":
        return ("hindari aktivitas di luar ruangan. Kalau terpaksa keluar, "
                "pakai masker N95/KN95.")
    return ("tetap di dalam ruangan, tutup pintu dan jendela, dan segera cari "
            "pertolongan medis jika sesak napas.")


@st.cache_resource
def muat_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def muat_data():
    daftar_file = sorted(DATA_RAW_DIR.glob(DATA_RAW_POLA))
    df = pd.concat([pd.read_csv(f) for f in daftar_file], ignore_index=True)

    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df["max"] = pd.to_numeric(df["max"], errors="coerce")
    df["stasiun"] = df["stasiun"].astype("string").str.strip()
    df["categori"] = df["categori"].astype("string").str.strip().str.upper()
    df["critical"] = df["critical"].astype("string").str.strip().str.upper()

    df = df.dropna(subset=["tanggal", "stasiun"])
    df = df.drop_duplicates(subset=["tanggal", "stasiun"], keep="first")
    tidak_ada_data = ((df["categori"] == "TIDAK ADA DATA")
                      | df["max"].isna() | (df["max"] <= 0))
    df = df[~tidak_ada_data]
    df = df[df["max"] <= ISPU_MAKS_WAJAR]
    df = df[df["max"].map(kategori_ispu) == df["categori"]]

    df = df.rename(columns={"max": "ispu", "critical": "polutan_kritis"})
    df["ispu"] = df["ispu"].astype(int)
    return (df[["tanggal", "stasiun", "ispu", "polutan_kritis"]]
            .sort_values(["stasiun", "tanggal"])
            .reset_index(drop=True))


def seri_stasiun(data, stasiun):
    tabel = data[data["stasiun"] == stasiun].set_index("tanggal").sort_index()
    return tabel.asfreq("D")


def hari_ini_terbaru(seri):
    ada = seri["ispu"].notna()
    return seri.index[ada & ada.shift(1, fill_value=False)].max()


def prediksi_besok(model, ispu_hari_ini, ispu_kemarin, tanggal):
    X_baru = pd.DataFrame(
        [[ispu_hari_ini, ispu_kemarin, tanggal.month, tanggal.day_of_week]],
        columns=FITUR,
    )
    hasil = float(model.predict(X_baru)[0])
    return min(max(hasil, 0.0), float(ISPU_MAKS_WAJAR))


def tgl_pendek(t):
    return f"{t.day} {NAMA_BULAN[t.month - 1]}"


def label_stasiun(nama):
    kode, _, lokasi = nama.partition(" (")
    return f"{kode} – {lokasi.rstrip(')')}" if lokasi else nama


def kartu(judul, isi, warna_teks="#2b2b2b", warna_latar="#f2efe9"):
    st.markdown(
        f'<div style="background:{warna_latar};color:{warna_teks};'
        f'border-radius:12px;padding:16px 20px">'
        f'<div style="font-size:0.95rem;opacity:0.85">{judul}</div>'
        f'<div style="font-size:2rem;font-weight:700;line-height:1.25">{isi}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def grafik(seri, hari_ini, nilai_prediksi):
    awal = hari_ini - pd.Timedelta(days=JUMLAH_HARI_GRAFIK - 1)
    besok = hari_ini + pd.Timedelta(days=1)

    asli = seri.loc[awal:hari_ini, ["ispu"]].reset_index()
    sambung = pd.DataFrame({
        "tanggal": [hari_ini, besok],
        "ispu": [seri.loc[hari_ini, "ispu"], nilai_prediksi],
    })
    titik_prediksi = pd.DataFrame({
        "tanggal": [besok], "ispu": [nilai_prediksi],
        "teks": [f"{round(nilai_prediksi)} (prediksi)"],
    })

    skala_y = alt.Scale(domain=[0, max(asli["ispu"].max(), nilai_prediksi,
                                       BATAS_TIDAK_SEHAT) * 1.2])
    x = alt.X("tanggal:T", title=None,
              axis=alt.Axis(format="%d %b", tickCount="day", grid=False),
              scale=alt.Scale(padding=30))
    y = alt.Y("ispu:Q", title="ISPU", scale=skala_y)

    garis_asli = alt.Chart(asli).mark_line(
        color="#3a86d9", strokeWidth=3,
        point=alt.OverlayMarkDef(color="#3a86d9", size=80),
    ).encode(x=x, y=y, tooltip=[alt.Tooltip("tanggal:T", format="%d %b %Y"),
                                alt.Tooltip("ispu:Q", title="ISPU")])
    garis_sambung = alt.Chart(sambung).mark_line(
        color="#e04545", strokeWidth=3, strokeDash=[6, 4]).encode(x=x, y=y)
    titik = alt.Chart(titik_prediksi).mark_point(
        color="#e04545", filled=True, size=220, opacity=1).encode(x=x, y=y)
    teks_prediksi = alt.Chart(titik_prediksi).mark_text(
        dy=-18, color="#555", fontSize=13).encode(x=x, y=y, text="teks:N")

    batas = pd.DataFrame({"tanggal": [awal], "ispu": [BATAS_TIDAK_SEHAT],
                          "teks": [f"batas tidak sehat ({BATAS_TIDAK_SEHAT})"]})
    garis_batas = alt.Chart(batas).mark_rule(
        color="#e04545", strokeDash=[5, 5], opacity=0.8).encode(y=y)
    teks_batas = alt.Chart(batas).mark_text(
        align="left", dy=-8, color="#888", fontSize=12).encode(x=x, y=y, text="teks:N")

    return alt.layer(garis_batas, teks_batas, garis_asli, garis_sambung,
                     titik, teks_prediksi).properties(height=320)


st.set_page_config(page_title="Prediksi Kualitas Udara Jakarta",
                   page_icon="🌫️", layout="centered")

model = muat_model()
data = muat_data()

st.title("Prediksi kualitas udara Jakarta")
st.caption("Berdasarkan data ISPU hari-hari sebelumnya")

kol_stasiun, kol_tombol = st.columns([4, 1.4], vertical_alignment="bottom")
with kol_stasiun:
    stasiun = st.selectbox("Stasiun pemantau", sorted(data["stasiun"].unique()),
                           format_func=label_stasiun, label_visibility="collapsed")
with kol_tombol:
    if st.button("Prediksi besok", type="primary", width="stretch"):
        st.session_state["stasiun_dipilih"] = stasiun

stasiun_dipilih = st.session_state.get("stasiun_dipilih")
if stasiun_dipilih is None:
    st.info("Pilih stasiun, lalu tekan **Prediksi besok**.")
    st.stop()

seri = seri_stasiun(data, stasiun_dipilih)
hari_ini = hari_ini_terbaru(seri)
kemarin = hari_ini - pd.Timedelta(days=1)
besok = hari_ini + pd.Timedelta(days=1)

ispu_hari_ini = int(seri.loc[hari_ini, "ispu"])
ispu_kemarin = int(seri.loc[kemarin, "ispu"])
polutan = str(seri.loc[hari_ini, "polutan_kritis"])
nilai = prediksi_besok(model, ispu_hari_ini, ispu_kemarin, hari_ini)
kat = kategori_ispu(round(nilai))
gaya = GAYA_KATEGORI[kat]

st.write("")
k1, k2, k3 = st.columns([1, 1.45, 1])
with k1:
    kartu(f"Hari ini ({tgl_pendek(hari_ini)})", ispu_hari_ini)
with k2:
    kartu(f"Prediksi besok ({tgl_pendek(besok)})",
          f"{round(nilai)} · {kat.capitalize()}",
          warna_teks=gaya["teks"], warna_latar=gaya["latar"])
with k3:
    kartu("Polutan kritis", NAMA_POLUTAN.get(polutan, polutan))

st.write("")
st.altair_chart(grafik(seri, hari_ini, nilai), width="stretch")

st.divider()
st.markdown(f"**Saran:** {saran(kat)}")

st.caption(f"Data terbaru di dataset: {tgl_pendek(hari_ini)} {hari_ini.year}. "
           "Angka besok adalah prediksi statistik dari data historis (rata-rata "
           "meleset ±11 poin), bukan informasi resmi pemerintah.")
