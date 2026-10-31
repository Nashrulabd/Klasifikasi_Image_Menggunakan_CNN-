import json, os
from pathlib import Path
import numpy as np, pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Klasifikasi Kehidupan Laut (CNN)", page_icon="🐠", layout="wide")

# Resolve paths relative to this app.py file, not the terminal's working directory.
BASE_DIR = Path(__file__).resolve().parent
_candidates = [BASE_DIR / "output", BASE_DIR / "outputs", BASE_DIR]
OUT = next((p for p in _candidates if (p / "class_names.json").exists()), BASE_DIR / "output")

REQUIRED_FILES = [
    "class_names.json", "metrics.json", "history.json",
    "class_distribution.csv", "classification_report.csv", "confusion_matrix.csv"
]
missing = [name for name in REQUIRED_FILES if not (OUT / name).is_file()]

st.title("🐠 Klasifikasi Citra Kehidupan Laut dengan CNN (MobileNetV2)")
if missing:
    st.error("Artefak hasil notebook belum ditemukan. Aplikasi tidak bisa menampilkan grafik dan evaluasi aktual sampai file berikut tersedia.")
    st.write(f"Folder artefak yang sedang diperiksa: `{OUT}`")
    st.markdown("**File yang belum ditemukan:**")
    for name in missing:
        st.write(f"- `{name}`")
    st.markdown("**Cara memperbaiki di Streamlit Cloud:**")
    st.markdown("1. Jalankan notebook sampai selesai agar file hasil eksperimen dibuat di folder `output/`.\n2. Upload/commit seluruh folder `output/` ke repository GitHub yang sama dengan `app.py`.\n3. Pastikan nama folder dan nama file sama persis (huruf besar/kecil juga berpengaruh).\n4. Commit dan push perubahan, lalu pilih **Reboot app** di Streamlit Cloud.")
    st.code("repository/\n├── app.py\n└── output/\n    ├── final_oceanic_cnn_model.keras\n    ├── class_names.json\n    ├── metrics.json\n    ├── history.json\n    ├── class_distribution.csv\n    ├── classification_report.csv\n    └── confusion_matrix.csv", language="text")
    st.info("File `app.py` saja tidak cukup: metrik, label kelas, dan model terlatih dibuat oleh notebook. Jangan mengganti hasil yang hilang dengan angka contoh karena dashboard harus menampilkan hasil eksperimen sebenarnya.")
    st.stop()

@st.cache_resource
def load_model():
    model_path = OUT / "final_oceanic_cnn_model.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}. Jalankan notebook hingga tahap penyimpanan model lalu unggah file tersebut ke repository.")
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_json(n):
    with (OUT / n).open(encoding="utf-8") as f:
        return json.load(f)

classes = load_json("class_names.json")
metrics = load_json("metrics.json")
hist = load_json("history.json")
dist = pd.read_csv(OUT / "class_distribution.csv", index_col=0)["jumlah"]
rep = pd.read_csv(OUT / "classification_report.csv", index_col=0)
cm = pd.read_csv(OUT / "confusion_matrix.csv", index_col=0)
IMG = tuple(metrics["img_size"])


st.write("Dataset *Oceanic Life* dari Kaggle. Dataset awal tidak memiliki label; label dibuat otomatis menggunakan **CLIP zero-shot**, kemudian model **MobileNetV2 (transfer learning)** dilatih. Kelas yang diprediksi adalah kategori kehidupan laut, **bukan emosi**: "
         f"**{len(classes)} kelas** — {', '.join(classes)}. Label evaluasi mengikuti hasil pelabelan CLIP, sehingga hasil juga bergantung pada kualitas label otomatis tersebut.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dataset", "📈 Training", "🎯 Evaluasi", "🔥 Grad-CAM & Contoh", "🔍 Prediksi Gambar Baru"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah kelas", len(classes)); c2.metric("Total gambar", int(dist.sum()))
    c3.metric("Data latih (80%)", metrics["n_train"]); c4.metric("Data uji (20%)", metrics["n_val"])
    st.subheader("Distribusi data per kelas"); st.bar_chart(dist)
    imb = dist.max() / dist.min()
    st.info(f"**Analisis:** kelas terbanyak **{dist.idxmax()}** ({dist.max()} gambar) dan tersedikit **{dist.idxmin()}** ({dist.min()} gambar); "
            f"rasio {imb:.1f}x. " + ("Terdapat ketidakseimbangan sehingga training memakai class weight." if imb > 1.5 else "Data relatif seimbang."))
    for f, cap in [("1_clip_label_audit.png", "Audit label CLIP: contoh gambar tiap kelas beserta keyakinan CLIP. Label dengan keyakinan < ambang dibuang agar data bersih."),
                   ("4_samples_per_class.png", "Contoh visual tiap kelas setelah resize; terlihat variasi warna, sudut, dan pencahayaan bawah laut."),
                   ("3_image_stats.png", "Statistik gambar: ukuran asli beragam (perlu resize), brightness cenderung rendah, ketajaman bervariasi.")]:
        if (OUT / f).exists(): st.image(str(OUT / f), caption=cap)

with tab2:
    st.subheader("Grafik akurasi & loss")
    st.caption("Garis accuracy/loss menunjukkan performa pada data latih dan validasi per epoch. Kurva validasi yang memburuk saat kurva latih terus membaik dapat menjadi indikasi overfitting.")
    h = pd.DataFrame(hist)
    c1, c2 = st.columns(2)
    c1.line_chart(h[["accuracy", "val_accuracy"]]); c2.line_chart(h[["loss", "val_loss"]])
    gap = hist["accuracy"][-1] - hist["val_accuracy"][-1]
    st.info(f"**Analisis:** akurasi latih akhir {hist['accuracy'][-1]:.3f} vs validasi {hist['val_accuracy'][-1]:.3f} (selisih {gap*100:.1f} poin). "
            + ("Ada indikasi overfitting." if gap > 0.1 else "Kurva konvergen dan model cukup general.")
            + f" Training: fase 1 = {metrics['epochs_phase1']} epoch, fase 2 (fine-tune) = {metrics['epochs_phase2']} epoch.")

with tab3:
    st.subheader("Hasil evaluasi model")
    c = st.columns(4)
    c[0].metric("Accuracy", f"{metrics['accuracy']:.3f}"); c[1].metric("Precision (macro)", f"{metrics['precision_macro']:.3f}")
    c[2].metric("Recall (macro)", f"{metrics['recall_macro']:.3f}"); c[3].metric("F1 (macro)", f"{metrics['f1_macro']:.3f}")
    st.dataframe(rep.style.format("{:.3f}"), use_container_width=True)
    per = rep.loc[classes]
    st.info(f"**Analisis:** kelas paling mudah dikenali **{per['f1-score'].idxmax()}** (F1 {per['f1-score'].max():.3f}); "
            f"paling sulit **{per['f1-score'].idxmin()}** (F1 {per['f1-score'].min():.3f}).")
    st.subheader("Confusion matrix")
    st.caption("Baris = label acuan dari CLIP; kolom = prediksi model. Nilai diagonal adalah prediksi yang cocok, sedangkan nilai di luar diagonal adalah pasangan kelas yang tertukar.")
    st.dataframe(cm.style.background_gradient(cmap="Blues"), use_container_width=True)
    if (OUT / "6_confusion_matrix.png").exists(): st.image(str(OUT / "6_confusion_matrix.png"))
    off = cm.values.copy(); np.fill_diagonal(off, 0); i, j = np.unravel_index(off.argmax(), off.shape)
    st.info(f"**Analisis:** kesalahan terbanyak: **{classes[i]}** diprediksi sebagai **{classes[j]}** ({off[i, j]} gambar), "
            "umumnya karena kemiripan visual atau label CLIP yang ambigu.")

with tab4:
    st.caption("Analisis gambar: contoh prediksi dibandingkan label CLIP. Grad-CAM adalah peta perhatian perkiraan model, bukan bukti kausal; area yang disorot perlu diperiksa apakah benar berada pada objek laut dan bukan latar.")
    for f, cap in [("7_prediction_samples.png", "Contoh prediksi vs label asli (hijau benar, merah salah)."),
                   ("8_gradcam.png", "Grad-CAM: area merah/kuning adalah bagian gambar yang paling memengaruhi keputusan model (objek utama = bagus; latar = model bisa terkecoh).")]:
        if (OUT / f).exists(): st.image(str(OUT / f), caption=cap)

def gradcam(model, arr):
    base = [l for l in model.layers if l.name.startswith("mobilenet")][0]
    feat = tf.keras.models.Model(base.input, base.output)
    x = tf.convert_to_tensor(arr[None].astype("float32")) / 255.0 * 2.0 - 1.0
    with tf.GradientTape() as tape:
        conv = feat(x, training=False); tape.watch(conv)
        p = model.get_layer("pred")(model.get_layer("drop")(model.get_layer("gap")(conv), training=False))
        k = int(tf.argmax(p[0])); score = p[:, k]
    g = tape.gradient(score, conv); w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, axis=-1))[0].numpy(); cam = cam / (cam.max() + 1e-8)
    return np.array(Image.fromarray((cam * 255).astype("uint8")).resize(IMG, Image.BILINEAR)) / 255.0

with tab5:
    st.subheader("Upload gambar baru")
    up = st.file_uploader("Pilih gambar (jpg/png)", type=["jpg", "jpeg", "png"])
    if up is not None:
        model = load_model()
        img = Image.open(up).convert("RGB"); arr = np.asarray(img.resize(IMG))
        probs = model.predict(arr[None].astype("float32"), verbose=0)[0]
        order = probs.argsort()[::-1]; top = order[0]
        c1, c2 = st.columns(2)
        c1.image(img, caption="Gambar yang diunggah", use_container_width=True)
        c2.success(f"### Prediksi: **{classes[top]}**  ({probs[top]*100:.1f}%)")
        c2.markdown("**Tiga kandidat teratas**")
        top_n = order[:min(3, len(order))]
        c2.dataframe(pd.DataFrame({"Kelas": [classes[i] for i in top_n], "Skor model (%)": [round(float(probs[i]*100), 2) for i in top_n]}), hide_index=True, use_container_width=True)
        c2.bar_chart(pd.Series(probs, index=classes))
        cam = gradcam(model, arr)
        import matplotlib.cm as cmap
        heat = (cmap.jet(cam)[..., :3] * 255).astype("uint8")
        over = (0.55 * arr + 0.45 * heat).astype("uint8")
        c1.image(over, caption="Grad-CAM (area yang memengaruhi prediksi)", use_container_width=True)
        conf = probs[top]; second = classes[order[1]]
        if conf >= 0.80:
            confidence_text = "Skor relatif tinggi, tetapi bukan jaminan prediksi benar."
        elif conf >= 0.55:
            confidence_text = "Skor menengah; periksa kemiripan dengan kandidat lain dan kualitas gambar."
        else:
            confidence_text = "Skor rendah; prediksi tidak pasti dan sebaiknya diverifikasi manual."
        st.info(f"**Analisis gambar:** model memilih **{classes[top]}** dengan skor {conf*100:.1f}%. {confidence_text} "
                + f"Kandidat kedua: {second} ({probs[order[1]]*100:.1f}%). Peta Grad-CAM menyoroti area yang berpengaruh pada keputusan model; jika sorotan berada di latar, prediksi mungkin dipengaruhi konteks dan bukan objek utama.")
