import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# TensorFlow opsional agar dashboard tidak langsung crash
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    TF_IMPORT_ERROR = None
except Exception as exc:
    tf = None
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(exc)

st.set_page_config(
    page_title="Klasifikasi Kehidupan Laut (CNN)",
    page_icon="🐠",
    layout="wide"
)

# Lokasi file
BASE_DIR = Path(__file__).resolve().parent
CANDIDATES = [
    BASE_DIR / "output",
    BASE_DIR / "outputs",
    BASE_DIR
]

OUT = next(
    (p for p in CANDIDATES if (p / "class_names.json").exists()),
    BASE_DIR
)

REQUIRED_FILES = [
    "class_names.json",
    "metrics.json",
    "history.json",
    "class_distribution.csv",
    "classification_report.csv",
    "confusion_matrix.csv"
]


def show_image(path, caption=None):
    """Buka gambar dengan PIL agar file rusak terdeteksi, lalu tampilkan."""
    try:
        with Image.open(path) as im:
            im.load()
            st.image(im.convert("RGB"), caption=caption, use_container_width=True)
    except Exception as exc:
        st.warning(f"Gambar `{Path(path).name}` tidak bisa dibuka: {exc}")

st.title("🐠 Klasifikasi Citra Kehidupan Laut dengan CNN")

# Periksa file hasil training
missing = [
    name for name in REQUIRED_FILES
    if not (OUT / name).is_file()
]

if missing:
    st.error("File hasil eksperimen belum lengkap.")

    st.write(f"Folder yang diperiksa: `{OUT}`")

    st.markdown("**File yang belum ditemukan:**")
    for name in missing:
        st.write(f"- `{name}`")

    st.info(
        "Pastikan seluruh file hasil notebook sudah diunggah "
        "ke GitHub dan berada di folder yang benar."
    )

    st.code(
        """repository/
├── app.py
├── requirements.txt
├── class_names.json
├── metrics.json
├── history.json
├── class_distribution.csv
├── classification_report.csv
├── confusion_matrix.csv
└── final_oceanic_cnn_model.keras""",
        language="text"
    )

    st.stop()


# Memuat model hanya saat dibutuhkan
@st.cache_resource
def load_model():
    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow belum terpasang. "
            f"Detail: {TF_IMPORT_ERROR}"
        )

    model_path = OUT / "final_oceanic_cnn_model.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}"
        )

    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_json(filename):
    with open(OUT / filename, encoding="utf-8") as file:
        return json.load(file)


# Memuat hasil eksperimen
classes = load_json("class_names.json")
metrics = load_json("metrics.json")
hist = load_json("history.json")

dist_df = pd.read_csv(
    OUT / "class_distribution.csv",
    index_col=0
)

dist = (
    dist_df["jumlah"]
    if "jumlah" in dist_df.columns
    else dist_df.iloc[:, 0]
)

report = pd.read_csv(
    OUT / "classification_report.csv",
    index_col=0
)

cm = pd.read_csv(
    OUT / "confusion_matrix.csv",
    index_col=0
)

IMG = tuple(metrics["img_size"])


st.write(
    "Dashboard klasifikasi dataset Oceanic Life menggunakan "
    "MobileNetV2 dan transfer learning. Label kelas dibuat "
    "menggunakan CLIP zero-shot. Hasil evaluasi bergantung "
    "pada kualitas label tersebut."
)

st.caption(
    f"Jumlah kelas: {len(classes)} — {', '.join(classes)}"
)

if not TF_AVAILABLE:
    st.warning(
        "TensorFlow belum tersedia. Dashboard evaluasi tetap "
        "bisa digunakan, tetapi prediksi gambar dan Grad-CAM "
        "belum bisa dijalankan."
    )


# Navigasi dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset",
    "📈 Training",
    "🎯 Evaluasi",
    "🔥 Grad-CAM & Contoh",
    "🔍 Prediksi Gambar Baru"
])


# ==================================================
# TAB 1: DATASET
# ==================================================
with tab1:
    st.subheader("Ringkasan dataset")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Jumlah kelas", len(classes))
    c2.metric("Total gambar", int(dist.sum()))
    c3.metric("Data latih", metrics["n_train"])
    c4.metric("Data validasi/uji", metrics["n_val"])

    st.subheader("Distribusi gambar per kelas")
    st.bar_chart(dist)

    if len(dist) and dist.min() > 0:
        imbalance = dist.max() / dist.min()

        st.info(
            f"Kelas terbanyak: {dist.idxmax()} "
            f"({dist.max()} gambar). "
            f"Kelas tersedikit: {dist.idxmin()} "
            f"({dist.min()} gambar). "
            f"Rasio ketidakseimbangan: {imbalance:.2f}x."
        )

    image_examples = [
        (
            "1_clip_label_audit.png",
            "Audit label otomatis CLIP"
        ),
        (
            "4_samples_per_class.png",
            "Contoh gambar tiap kelas"
        ),
        (
            "3_image_stats.png",
            "Statistik gambar dataset"
        )
    ]

    for filename, caption in image_examples:
        image_path = OUT / filename

        if image_path.exists():
            show_image(image_path, caption)


# ==================================================
# TAB 2: TRAINING
# ==================================================
with tab2:
    st.subheader("Grafik akurasi dan loss")

    history_df = pd.DataFrame(hist)

    available_accuracy = [
        col for col in ["accuracy", "val_accuracy"]
        if col in history_df.columns
    ]

    available_loss = [
        col for col in ["loss", "val_loss"]
        if col in history_df.columns
    ]

    col1, col2 = st.columns(2)

    if available_accuracy:
        col1.markdown("**Akurasi per epoch**")
        col1.line_chart(history_df[available_accuracy])
    else:
        col1.warning("Data akurasi tidak tersedia.")

    if available_loss:
        col2.markdown("**Loss per epoch**")
        col2.line_chart(history_df[available_loss])
    else:
        col2.warning("Data loss tidak tersedia.")

    if (
        "accuracy" in history_df.columns
        and "val_accuracy" in history_df.columns
        and len(history_df) > 0
    ):
        train_acc = float(history_df["accuracy"].iloc[-1])
        val_acc = float(history_df["val_accuracy"].iloc[-1])
        gap = train_acc - val_acc

        st.info(
            f"Akurasi training akhir: {train_acc:.3f}. "
            f"Akurasi validasi akhir: {val_acc:.3f}. "
            f"Selisih: {gap * 100:.1f} poin persentase. "
            + (
                "Ada indikasi overfitting."
                if gap > 0.1
                else "Periksa kedua kurva untuk menilai generalisasi model."
            )
        )

    phase1 = metrics.get("epochs_phase1", "-")
    phase2 = metrics.get("epochs_phase2", "-")

    st.write(f"Epoch fase pertama: {phase1}")
    st.write(f"Epoch fine-tuning: {phase2}")


# ==================================================
# TAB 3: EVALUASI
# ==================================================
with tab3:
    st.subheader("Metrik evaluasi model")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.3f}"
    )
    c2.metric(
        "Precision Macro",
        f"{metrics['precision_macro']:.3f}"
    )
    c3.metric(
        "Recall Macro",
        f"{metrics['recall_macro']:.3f}"
    )
    c4.metric(
        "F1-Score Macro",
        f"{metrics['f1_macro']:.3f}"
    )

    st.subheader("Classification report")
    st.dataframe(
        report.style.format("{:.3f}"),
        use_container_width=True
    )

    if "f1-score" in report.columns:
        per_class = report.reindex(classes).dropna(
            subset=["f1-score"]
        )

        if not per_class.empty:
            best = per_class["f1-score"].idxmax()
            worst = per_class["f1-score"].idxmin()

            st.info(
                f"F1 tertinggi: {best} "
                f"({per_class.loc[best, 'f1-score']:.3f}). "
                f"F1 terendah: {worst} "
                f"({per_class.loc[worst, 'f1-score']:.3f})."
            )

    st.subheader("Confusion matrix")
    st.caption(
        "Baris menunjukkan label acuan dan kolom menunjukkan "
        "prediksi. Nilai diagonal adalah prediksi yang cocok."
    )

    st.dataframe(
        cm.style.background_gradient(cmap="Blues"),
        use_container_width=True
    )

    cm_image = OUT / "6_confusion_matrix.png"

    if cm_image.exists():
        show_image(cm_image)

    if cm.shape[0] > 1 and cm.shape[1] > 1:
        values = cm.to_numpy().copy()
        np.fill_diagonal(values, 0)

        i, j = np.unravel_index(
            np.argmax(values),
            values.shape
        )

        st.info(
            f"Kesalahan terbanyak: kelas {cm.index[i]} "
            f"diprediksi sebagai {cm.columns[j]} "
            f"sebanyak {int(values[i, j])} gambar."
        )


# ==================================================
# TAB 4: GRAD-CAM DAN CONTOH
# ==================================================
with tab4:
    st.subheader("Contoh hasil klasifikasi")

    st.caption(
        "Grad-CAM merupakan visualisasi area yang berpengaruh "
        "pada prediksi model, bukan bukti sebab-akibat."
    )

    examples = [
        (
            "7_prediction_samples.png",
            "Contoh hasil prediksi"
        ),
        (
            "8_gradcam.png",
            "Contoh visualisasi Grad-CAM"
        )
    ]

    for filename, caption in examples:
        image_path = OUT / filename

        if image_path.exists():
            show_image(image_path, caption)

    if not any((OUT / f).exists() for f, _ in examples):
        st.info(
            "Gambar contoh belum tersedia. "
            "Unggah artefak visual dari notebook jika diperlukan."
        )


# ==================================================
# FUNGSI GRAD-CAM
# ==================================================
def gradcam(model, arr):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow tidak tersedia.")

    base_layers = [
        layer for layer in model.layers
        if layer.name.startswith("mobilenet")
    ]

    if not base_layers:
        raise ValueError(
            "Layer MobileNet tidak ditemukan pada model."
        )

    base = base_layers[0]
    feature_model = tf.keras.Model(
        inputs=base.input,
        outputs=base.output
    )

    x = tf.convert_to_tensor(
        arr[None].astype("float32") / 127.5 - 1.0
    )

    with tf.GradientTape() as tape:
        conv = feature_model(x, training=False)
        tape.watch(conv)

        gap_layer = model.get_layer("gap")
        drop_layer = model.get_layer("drop")
        pred_layer = model.get_layer("pred")

        pooled = gap_layer(conv)
        dropped = drop_layer(pooled, training=False)
        predictions = pred_layer(dropped)

        class_index = int(tf.argmax(predictions[0]))
        score = predictions[:, class_index]

    gradients = tape.gradient(score, conv)
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
        keepdims=True
    )

    cam = tf.nn.relu(
        tf.reduce_sum(weights * conv, axis=-1)
    )[0].numpy()

    cam = cam / (cam.max() + 1e-8)

    cam_image = Image.fromarray(
        (cam * 255).astype("uint8")
    ).resize(
        (IMG[1], IMG[0]),
        Image.Resampling.BILINEAR
    )

    return np.asarray(cam_image) / 255.0


# ==================================================
# TAB 5: PREDIKSI GAMBAR BARU
# ==================================================
with tab5:
    st.subheader("Prediksi gambar kehidupan laut")

    if not TF_AVAILABLE:
        st.warning(
            "Prediksi belum dapat dijalankan karena TensorFlow "
            "tidak terpasang di server."
        )

        st.code(
            f"Detail error: {TF_IMPORT_ERROR}",
            language="text"
        )

        st.caption(
            "Pastikan requirements.txt memuat tensorflow-cpu "
            "dan aplikasi berjalan pada Python yang kompatibel."
        )

    uploaded_file = st.file_uploader(
        "Unggah gambar JPG atau PNG",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None and TF_AVAILABLE:
        try:
            model = load_model()

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            arr = np.asarray(
                image.resize((IMG[1], IMG[0]))
            )

            probabilities = model.predict(
                arr[None].astype("float32"),
                verbose=0
            )[0]

            order = probabilities.argsort()[::-1]
            top = int(order[0])

            col1, col2 = st.columns(2)

            col1.image(
                image,
                caption="Gambar yang diunggah",
                use_container_width=True
            )

            col2.success(
                f"Prediksi: {classes[top]} "
                f"({probabilities[top] * 100:.1f}%)"
            )

            top_n = order[:min(3, len(order))]

            top_df = pd.DataFrame({
                "Kelas": [classes[i] for i in top_n],
                "Skor model (%)": [
                    round(float(probabilities[i] * 100), 2)
                    for i in top_n
                ]
            })

            col2.markdown("**Tiga kandidat teratas**")
            col2.dataframe(
                top_df,
                hide_index=True,
                use_container_width=True
            )

            col2.bar_chart(
                pd.Series(
                    probabilities,
                    index=classes
                )
            )

            try:
                cam = gradcam(model, arr)

                import matplotlib.cm as cmap

                heat = (
                    cmap.jet(cam)[..., :3] * 255
                ).astype("uint8")

                overlay = (
                    0.55 * arr + 0.45 * heat
                ).clip(0, 255).astype("uint8")

                col1.image(
                    overlay,
                    caption="Visualisasi Grad-CAM",
                    use_container_width=True
                )

            except Exception as exc:
                st.warning(
                    f"Grad-CAM tidak dapat dibuat: {exc}"
                )

            confidence = float(probabilities[top])

            if confidence >= 0.80:
                confidence_text = (
                    "Skor relatif tinggi, tetapi bukan jaminan benar."
                )
            elif confidence >= 0.55:
                confidence_text = (
                    "Skor menengah; periksa kandidat lainnya."
                )
            else:
                confidence_text = (
                    "Skor rendah; prediksi belum meyakinkan."
                )

            second = int(order[1]) if len(order) > 1 else top

            st.info(
                f"Analisis: model memilih **{classes[top]}** "
                f"dengan skor {confidence * 100:.1f}%. "
                f"{confidence_text} "
                f"Kandidat berikutnya: {classes[second]} "
                f"({probabilities[second] * 100:.1f}%)."
            )

        except Exception as exc:
            st.error(f"Prediksi gagal dijalankan: {exc}")
