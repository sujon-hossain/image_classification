import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageEnhance, UnidentifiedImageError
from torchvision import models, transforms

# --------------------------------------------------------------------------
# Page Config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="VisionLab Studio • ResNet-18",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CLASS_ICONS = {
    "airplane": "✈️", "automobile": "🚗", "bird": "🐦", "cat": "🐱", "deer": "🦌",
    "dog": "🐶", "frog": "🐸", "horse": "🐴", "ship": "🚢", "truck": "🚚",
}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

MODELS = {
    "Fine-tuned": {
        "file": BASE_DIR/"best_a.pt",
        "color": "#06B6D4",        # Electric Cyan
        "gradient": "linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%)",
        "badge_bg": "rgba(6, 182, 212, 0.14)",
        "badge": "Full Network Adapted",
        "note": "All convolutional blocks & classifier head fine-tuned on CIFAR-10",
    },
    "Frozen": {
        "file":  BASE_DIR/"best_b.pt",
        "color": "#F43F5E",        # Vibrant Rose
        "gradient": "linear-gradient(135deg, #F43F5E 0%, #F97316 100%)",
        "badge_bg": "rgba(244, 63, 94, 0.14)",
        "badge": "Linear Probe Only",
        "note": "ImageNet feature extractor frozen; only final linear layer trained",
    },
}

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


@st.cache_resource
def load_model(path_str: str):
    path = Path(path_str)
    has_weights = path.exists()
    m = models.resnet18(weights=None if has_weights else models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    if has_weights:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        m.load_state_dict(state_dict)
    m.eval()
    return m, has_weights


def predict(model, image: Image.Image) -> tuple[np.ndarray, float]:
    x = transform(image).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    latency_ms = (time.perf_counter() - t0) * 1000
    return probs, latency_ms


def make_sample_image(kind: str) -> Image.Image:
    img = Image.new("RGB", (320, 320), "#E0F2FE")
    draw = ImageDraw.Draw(img)
    if kind == "✈️ Airplane":
        draw.rectangle([0, 0, 320, 320], fill="#0284C7")
        draw.ellipse([210, 35, 275, 100], fill="#FDE047")
        draw.polygon([(45, 175), (275, 145), (225, 185), (45, 195)], fill="#F8FAFC")
        draw.polygon([(135, 162), (180, 95), (195, 155)], fill="#E2E8F0")
    elif kind == "🚢 Cargo Ship":
        draw.rectangle([0, 0, 320, 185], fill="#7DD3FC")
        draw.rectangle([0, 185, 320, 320], fill="#0369A1")
        draw.polygon([(60, 185), (260, 185), (235, 230), (85, 230)], fill="#0F172A")
        draw.rectangle([105, 140, 215, 185], fill="#F8FAFC")
        draw.rectangle([135, 105, 158, 140], fill="#F43F5E")
    else:  # 🚗 Sports Car
        draw.rectangle([0, 0, 320, 215], fill="#FEF3C7")
        draw.rectangle([0, 215, 320, 320], fill="#1E293B")
        draw.rounded_rectangle([55, 155, 265, 215], radius=18, fill="#E11D48")
        draw.rounded_rectangle([95, 115, 225, 168], radius=16, fill="#FB7185")
        draw.ellipse([80, 195, 122, 237], fill="#090D16")
        draw.ellipse([198, 195, 240, 237], fill="#090D16")
    return img


# --------------------------------------------------------------------------
# Styling (Modern Studio Theme)
# --------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"], .stApp {
  font-family: 'Inter', sans-serif;
  background-color: #0B1120;
  color: #F1F5F9;
}

#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1260px; }

section[data-testid="stSidebar"] {
  background-color: #0F172A;
  border-right: 1px solid #1E293B;
}

.studio-header {
  background: radial-gradient(circle at top right, rgba(6, 182, 212, 0.22), transparent 45%),
              radial-gradient(circle at bottom left, rgba(244, 63, 94, 0.18), transparent 45%),
              #111827;
  border: 1px solid #1F2937;
  border-radius: 22px;
  padding: 1.8rem 2.2rem;
  margin-bottom: 1.2rem;
}
.studio-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  font-weight: 700;
  color: #38BDF8;
  background: rgba(56, 189, 248, 0.12);
  border: 1px solid rgba(56, 189, 248, 0.3);
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  margin-bottom: 0.6rem;
}
.studio-title {
  font-size: clamp(1.75rem, 3.5vw, 2.5rem);
  font-weight: 800;
  letter-spacing: -0.03em;
  margin: 0;
  background: linear-gradient(90deg, #FFFFFF 0%, #94A3B8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.studio-sub {
  color: #94A3B8;
  font-size: 0.98rem;
  margin: 0.35rem 0 0 0;
}

.chips { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-bottom: 1.3rem; }
.chip {
  font-size: 0.84rem;
  font-weight: 500;
  padding: 0.3rem 0.8rem;
  border-radius: 999px;
  background: #111827;
  border: 1px solid #1F2937;
  color: #94A3B8;
  transition: all 0.25s ease;
}
.chip.on {
  background: linear-gradient(135deg, #06B6D4, #3B82F6);
  color: #FFFFFF;
  border-color: transparent;
  font-weight: 700;
  box-shadow: 0 0 18px rgba(6, 182, 212, 0.45);
  transform: translateY(-2px);
}

[data-testid="stFileUploader"] section {
  background: #111827;
  border: 1.5px dashed #334155;
  border-radius: 16px;
  padding: 1.3rem;
}
[data-testid="stFileUploader"] section:hover {
  border-color: #06B6D4;
  background: #0F172A;
}

[data-testid="stImage"] img {
  border-radius: 16px;
  border: 1px solid #1E293B;
  background: #111827;
  padding: 6px;
}

.pred-card {
  background: #111827;
  border: 1px solid #1F2937;
  border-radius: 18px;
  padding: 1.4rem;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s ease, border-color 0.2s ease;
}
.pred-card:hover {
  transform: translateY(-3px);
  border-color: var(--accent);
}
.pred-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 4px;
  background: var(--grad);
}
.card-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.3rem;
}
.card-title { font-weight: 800; font-size: 1.15rem; }
.card-badge {
  font-size: 0.72rem;
  font-weight: 700;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
}
.card-desc { color: #64748B; font-size: 0.84rem; margin-bottom: 1rem; }
.card-class {
  font-size: 2.1rem;
  font-weight: 800;
  text-transform: capitalize;
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.card-meta {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  color: #94A3B8;
  margin: 0.3rem 0 1.1rem 0;
}

.p-row {
  display: grid;
  grid-template-columns: 6.5rem 1fr 3.6rem;
  align-items: center;
  gap: 0.6rem;
  margin: 0.42rem 0;
}
.p-lbl { font-size: 0.88rem; text-transform: capitalize; color: #94A3B8; }
.p-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  text-align: right;
  color: #94A3B8;
}
.p-track { height: 8px; background: #1E293B; border-radius: 999px; overflow: hidden; }
.p-fill { height: 100%; border-radius: 999px; transition: width 0.5s ease; }
.p-row.best .p-lbl, .p-row.best .p-val { color: #F8FAFC; font-weight: 700; }

.banner {
  border-radius: 14px;
  padding: 0.85rem 1.2rem;
  margin-bottom: 1rem;
  font-size: 0.96rem;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.banner.match {
  background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.35);
  color: #6EE7B7;
}
.banner.split {
  background: rgba(244, 63, 94, 0.12);
  border: 1px solid rgba(244, 63, 94, 0.35);
  color: #FDA4AF;
}

.empty-panel {
  background: #111827;
  border: 1px dashed #334155;
  border-radius: 18px;
  padding: 3.5rem 2rem;
  text-align: center;
  color: #94A3B8;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Sidebar Controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎛️ VisionLab Controls")
    view_mode = st.radio(
        "Active Architecture",
        ["Compare Both", "Fine-tuned Only", "Frozen Only"],
    )
    top_k = st.slider("Top Classes in Card", 3, 10, 5)
    chart_style = st.selectbox("Chart Visualization", ["Bar Chart", "Area Chart", "Off"])

    st.divider()
    st.markdown("### 🖼️ Image Pre-Adjustments")
    brightness = st.slider("Brightness", 0.5, 1.5, 1.0, 0.05)
    contrast = st.slider("Contrast", 0.5, 1.5, 1.0, 0.05)

    st.divider()
    st.markdown("### 🧠 Weights Status")
    for m_name, info in MODELS.items():
        ok = info["file"].exists()
        st.caption(f"{'🟢' if ok else '🟡'} **{m_name}**: `{info['file'].name}` {'loaded' if ok else '(fallback mode)'}")

# --------------------------------------------------------------------------
# Header & Live Class Chips
# --------------------------------------------------------------------------
st.markdown("""
<div class="studio-header">
  <div>
    <div class="studio-pill">✦ RESNET-18 TRANSFER LEARNING BENCHMARK</div>
    <h1 class="studio-title">What’s in this picture?</h1>
    <p class="studio-sub">Real-time probability breakdown comparing full fine-tuning vs. frozen feature extraction.</p>
  </div>
</div>
""", unsafe_allow_html=True)

chips_slot = st.empty()

# --------------------------------------------------------------------------
# Main Workspace
# --------------------------------------------------------------------------
left, right = st.columns([5, 7], gap="large")

with left:
    t_upload, t_sample, t_cam = st.tabs(["📁 Upload Image", "🎨 Quick Samples", "📷 Webcam"])
    raw_image = None

    with t_upload:
        up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if up is not None:
            try:
                raw_image = Image.open(up).convert("RGB")
            except (UnidentifiedImageError, OSError):
                st.error("Invalid image file. Please upload a valid JPG or PNG.")

    with t_sample:
        sample = st.radio("Choose sample:", ["None", "✈️ Airplane", "🚢 Cargo Ship", "🚗 Sports Car"], horizontal=True)
        if sample != "None" and raw_image is None:
            raw_image = make_sample_image(sample)

    with t_cam:
        snap = st.camera_input("Capture", label_visibility="collapsed")
        if snap is not None:
            raw_image = Image.open(snap).convert("RGB")

    image = None
    if raw_image is not None:
        image = ImageEnhance.Brightness(raw_image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        st.image(image, caption=f"Input Tensor: 128×128 RGB • Source: {image.width}×{image.height}px", width="stretch")


def build_bars(probs: np.ndarray, color: str, k: int) -> str:
    order = np.argsort(-probs)[:k]
    out = []
    for rank, idx in enumerate(order):
        pct = probs[idx] * 100
        op = "1.0" if rank == 0 else "0.4"
        cls = "p-row best" if rank == 0 else "p-row"
        out.append(
            f'<div class="{cls}">'
            f'<span class="p-lbl">{CLASS_ICONS[CLASSES[idx]]} {CLASSES[idx]}</span>'
            f'<div class="p-track"><div class="p-fill" style="width:{max(pct, 1.0):.1f}%;background:{color};opacity:{op}"></div></div>'
            f'<span class="p-val">{pct:.1f}%</span>'
            f'</div>'
        )
    return "".join(out)


def build_card(name: str, info: dict, probs: np.ndarray, latency: float, k: int) -> str:
    top = int(np.argmax(probs))
    return (
        f'<div class="pred-card" style="--accent:{info["color"]};--grad:{info["gradient"]}">'
        f'<div class="card-top">'
        f'<span class="card-title" style="color:{info["color"]}">{name}</span>'
        f'<span class="card-badge" style="background:{info["badge_bg"]};color:{info["color"]}">{info["badge"]}</span>'
        f'</div>'
        f'<div class="card-desc">{info["note"]}</div>'
        f'<div class="card-class">{CLASS_ICONS[CLASSES[top]]} {CLASSES[top]}</div>'
        f'<div class="card-meta">{probs[top]*100:.1f}% confidence • {latency:.1f} ms</div>'
        f'{build_bars(probs, info["color"], k)}'
        f'</div>'
    )


predicted_set = set()

with right:
    if image is None:
        st.markdown("""
        <div class="empty-panel">
          <h3>✦ Ready for Inference</h3>
          <p>Upload a photo, use your webcam, or click a <b>Quick Sample</b> on the left to inspect live model predictions and probability distributions.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        selected = (
            ["Fine-tuned"] if view_mode == "Fine-tuned Only"
            else ["Frozen"] if view_mode == "Frozen Only"
            else ["Fine-tuned", "Frozen"]
        )

        results, latencies = {}, {}
        for m_name in selected:
            m_obj, _ = load_model(str(MODELS[m_name]["file"]))
            results[m_name], latencies[m_name] = predict(m_obj, image)

        tops = {m_name: int(np.argmax(p)) for m_name, p in results.items()}
        predicted_set = set(tops.values())

        if len(selected) == 2:
            a, b = tops["Fine-tuned"], tops["Frozen"]
            delta = (results["Fine-tuned"][a] - results["Frozen"][b]) * 100
            if a == b:
                st.markdown(
                    f'<div class="banner match"><span>✅ <b>Models Agree:</b> Both predict <b>{CLASSES[a].title()}</b></span>'
                    f'<span>Δ Confidence: <b>{delta:+.1f}%</b></span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="banner split"><span>⚡ <b>Disagreement:</b> Fine-tuned → <b>{CLASSES[a].title()}</b> | Frozen → <b>{CLASSES[b].title()}</b></span></div>',
                    unsafe_allow_html=True,
                )

        cols = st.columns(len(selected), gap="medium")
        for col, m_name in zip(cols, selected):
            with col:
                st.markdown(
                    build_card(m_name, MODELS[m_name], results[m_name], latencies[m_name], top_k),
                    unsafe_allow_html=True,
                )

        if chart_style != "Off":
            st.markdown("#### 📊 Confidence Distribution (%)")
            chart_data = {"Class": [f"{CLASS_ICONS[c]} {c.title()}" for c in CLASSES]}
            chart_colors = []
            for m_name in selected:
                chart_data[m_name] = np.round(results[m_name] * 100, 2)
                chart_colors.append(MODELS[m_name]["color"])

            df = pd.DataFrame(chart_data).set_index("Class")
            if chart_style == "Bar Chart":
                st.bar_chart(df, color=chart_colors, height=280)
            else:
                st.area_chart(df, color=chart_colors, height=280)

# Top Category Chips
chips_html = "".join(
    f'<span class="chip{" on" if i in predicted_set else ""}">{CLASS_ICONS[c]} {c}</span>'
    for i, c in enumerate(CLASSES)
)
chips_slot.markdown(f'<div class="chips">{chips_html}</div>', unsafe_allow_html=True)
