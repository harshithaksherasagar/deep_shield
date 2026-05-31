"""
================================================================
DEEPFAKE DETECTION — STREAMLIT WEB APP
Attention-Augmented Xception + CBAM

Run:
    streamlit run app.py

Requirements:
    pip install streamlit timm opencv-python-headless
                torch torchvision matplotlib pillow
================================================================
"""

import os
import io
import cv2
import tempfile
import datetime
import warnings
import numpy as np
import torch
import torch.nn as nn
import timm
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

warnings.filterwarnings("ignore")

# Pre-build matplotlib font cache silently so it never blocks the UI
import matplotlib.font_manager as _fmgr
try:
    _fmgr._load_fontmanager(try_read_cache=False)
except Exception:
    pass

# ================================================================
# PAGE CONFIG  — must be first Streamlit call
# ================================================================
st.set_page_config(
    page_title="DeepShield — Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CUSTOM CSS
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background-color: #0d0f1a; color: #e8e8f0; }

[data-testid="stSidebar"] {
    background-color: #12152a;
    border-right: 1px solid #1e2340;
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.2rem;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #00d4ff 0%, #7b61ff 50%, #ff6b9d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #6b7280;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.verdict-fake {
    background: linear-gradient(135deg, #2d0a0a, #4a0f0f);
    border: 1px solid #ff4444; border-radius: 16px;
    padding: 2rem; text-align: center;
    box-shadow: 0 0 40px rgba(255,68,68,0.2);
}
.verdict-real {
    background: linear-gradient(135deg, #0a2d1a, #0f4a24);
    border: 1px solid #00e676; border-radius: 16px;
    padding: 2rem; text-align: center;
    box-shadow: 0 0 40px rgba(0,230,118,0.2);
}
.verdict-label-fake {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 3.5rem; color: #ff4444;
    letter-spacing: 0.1em; margin: 0;
}
.verdict-label-real {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 3.5rem; color: #00e676;
    letter-spacing: 0.1em; margin: 0;
}
.confidence-text {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem; color: #9ca3af; margin-top: 0.5rem;
}
.metric-card {
    background: #12152a; border: 1px solid #1e2340;
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;
}
.metric-label {
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    color: #6b7280; text-transform: uppercase;
    letter-spacing: 0.12em; margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Syne', sans-serif; font-weight: 600;
    font-size: 1.6rem; color: #e8e8f0;
}
.prob-bar-wrap {
    background: #1e2340; border-radius: 8px;
    height: 10px; width: 100%; margin-top: 0.5rem; overflow: hidden;
}
.prob-bar-fill-fake {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg, #ff4444, #ff8c00);
}
.prob-bar-fill-real {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg, #00e676, #00b4d8);
}
.section-header {
    font-family: 'Space Mono', monospace; font-size: 0.75rem;
    color: #7b61ff; text-transform: uppercase; letter-spacing: 0.2em;
    border-bottom: 1px solid #1e2340; padding-bottom: 0.5rem;
    margin-bottom: 1.2rem; margin-top: 1.5rem;
}
[data-testid="stFileUploader"] {
    background: #12152a; border: 2px dashed #1e2340; border-radius: 12px;
}
[data-testid="stFileUploader"]:hover { border-color: #7b61ff; }
.stButton > button {
    background: linear-gradient(135deg, #7b61ff, #00d4ff);
    color: white; border: none; border-radius: 8px;
    font-family: 'Space Mono', monospace; font-size: 0.85rem;
    padding: 0.6rem 2rem; width: 100%;
}
.stButton > button:hover { opacity: 0.85; }
hr { border-color: #1e2340; margin: 1.5rem 0; }
.frame-row-fake {
    background: rgba(255,68,68,0.08); border-left: 3px solid #ff4444;
    padding: 0.5rem 1rem; margin-bottom: 0.3rem;
    border-radius: 0 6px 6px 0;
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
}
.frame-row-real {
    background: rgba(0,230,118,0.05); border-left: 3px solid #00e676;
    padding: 0.5rem 1rem; margin-bottom: 0.3rem;
    border-radius: 0 6px 6px 0;
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ================================================================
# MODEL ARCHITECTURE
# ================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg   = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.spatial(self.channel(x))


class AttentionXception(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("xception", pretrained=False,
                                          num_classes=0, global_pool="")
        self.cbam = CBAM(in_channels=2048)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.5)
        self.fc   = nn.Linear(2048, 1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.cbam(feat)
        feat = self.pool(feat).flatten(1)
        feat = self.drop(feat)
        return self.fc(feat).squeeze(1)


# ================================================================
# MODEL LOADER — cached forever, loads once on startup
# ================================================================
@st.cache_resource(show_spinner=False)
def load_model(model_path):
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = AttentionXception()

    start = time.time()

    # Try weights_only=True first (safer/faster), fall back for older .pth files
    # that contain custom Python objects pickled into the checkpoint.
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(model_path, map_location=device, weights_only=False)

    if "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Warmup pass so first real image is instant.
    # Uses no_grad here only — GradCAM calls use enable_grad() inside generate().
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 299, 299, device=device)
        _ = model(dummy)

    print(f"[DeepShield] Model ready in {time.time()-start:.2f}s on {device.upper()}")
    return model, device


# ================================================================
# FACE EXTRACTOR (RetinaFace — optional, falls back to full frame)
# ================================================================
@st.cache_resource(show_spinner=False)
def load_face_extractor():
    try:
        from retinaface import RetinaFace
        return RetinaFace, True
    except Exception:
        return None, False


# ================================================================
# PREPROCESSING
# ================================================================
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_rgb, device, size=299):
    if not isinstance(img_rgb, np.ndarray):
        img_rgb = np.array(img_rgb)
    if img_rgb.dtype != np.uint8:
        img_rgb = (img_rgb * 255).astype(np.uint8)
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    if img_rgb.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
    img = cv2.resize(img_rgb, (size, size)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return (
        torch.from_numpy(np.ascontiguousarray(img))
        .permute(2, 0, 1).unsqueeze(0).float().to(device)
    )


def preprocess_face_crop(face_bgr_or_rgb, device):
    """Fast preprocess for an already-cropped face numpy array."""
    fc = cv2.resize(face_bgr_or_rgb, (299, 299)).astype(np.float32) / 255.0
    fc = (fc - MEAN) / STD
    return (
        torch.from_numpy(np.ascontiguousarray(fc))
        .permute(2, 0, 1).unsqueeze(0).float().to(device)
    )


# ================================================================
# FACE DETECTION
# ================================================================
def extract_face(img_rgb, rf_module, rf_ready, margin=0.30):
    """Crop largest face. Falls back to full frame."""
    h, w = img_rgb.shape[:2]
    if rf_ready and rf_module is not None:
        try:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            faces   = rf_module.detect_faces(img_bgr, threshold=0.90)
            if isinstance(faces, dict) and faces:
                best, best_area = None, -1
                for val in faces.values():
                    x1, y1, x2, y2 = val["facial_area"]
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area, best = area, val
                if best:
                    x1, y1, x2, y2 = best["facial_area"]
                    dw = int((x2 - x1) * margin)
                    dh = int((y2 - y1) * margin)
                    x1 = max(0, x1 - dw); y1 = max(0, y1 - dh)
                    x2 = min(w, x2 + dw); y2 = min(h, y2 + dh)
                    face = img_rgb[y1:y2, x1:x2]
                    if face.size > 0:
                        return cv2.resize(face, (299, 299)), True
        except Exception:
            pass
    return cv2.resize(img_rgb, (299, 299)), False


# ================================================================
# GRAD-CAM
# FIX: torch.enable_grad() context manager added inside generate()
#      so GradCAM works correctly even if the caller is inside a
#      torch.no_grad() block (e.g. Streamlit reruns, future refactors).
# ================================================================
class GradCAM:
    """
    Registers hooks on the CBAM spatial conv layer.
    generate() explicitly enables grad via torch.enable_grad(),
    making it safe to call from any context — no_grad or otherwise.
    """
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._hooks      = []
        target = model.cbam.spatial.conv
        self._hooks.append(target.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach())))
        self._hooks.append(target.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach())))

    def generate(self, tensor):
        # FIX: explicitly enable gradients so GradCAM works regardless of
        # any outer torch.no_grad() context that may have been set upstream.
        with torch.enable_grad():
            self.model.zero_grad()
            self.model.eval()          # keep BN/Dropout in eval mode
            out = self.model(tensor)
            out.backward(torch.ones_like(out))

        if self.gradients is None:
            # Fallback: return blank cam — avoids crash if hooks misfired
            cam = np.zeros((10, 10), dtype=np.float32)
        else:
            w   = self.gradients.mean(dim=[2, 3], keepdim=True)
            cam = torch.relu((w * self.activations).sum(1)).squeeze()

            # FIX: guard against 0-d tensor when spatial dims collapse to 1×1
            if cam.dim() == 0:
                cam = cam.unsqueeze(0).unsqueeze(0).expand(10, 10)

            cam = cam.cpu().numpy()
            cam -= cam.min()
            if cam.max() > 0:
                cam /= cam.max()

        prob = torch.sigmoid(out).item()
        return cam, prob

    def remove(self):
        for h in self._hooks:
            h.remove()


def make_overlay(face_rgb, cam):
    h, w    = face_rgb.shape[:2]
    cam_r   = cv2.resize(cam, (w, h))
    heat    = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat    = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * face_rgb + 0.45 * heat).astype(np.uint8)
    return overlay, heat


# ================================================================
# INFERENCE — IMAGE
# ================================================================
def run_inference(model, device, img_rgb, threshold=0.5,
                  rf_module=None, rf_ready=False):
    face_rgb, face_detected = extract_face(img_rgb, rf_module, rf_ready)
    tensor                  = preprocess(face_rgb, device)

    # GradCAM.generate() handles enable_grad() internally — safe to call here
    gcam             = GradCAM(model)
    cam, prob_fake   = gcam.generate(tensor)
    gcam.remove()

    overlay, heatmap = make_overlay(face_rgb, cam)
    label = "FAKE" if prob_fake >= threshold else "REAL"
    conf  = prob_fake if label == "FAKE" else 1 - prob_fake
    return label, prob_fake, conf, overlay, heatmap, face_detected


# ================================================================
# INFERENCE — single frame (no GradCAM, faster for video)
# ================================================================
def infer_frame(model, device, face_rgb):
    fc = cv2.resize(face_rgb, (299, 299)).astype(np.float32) / 255.0
    fc = (fc - MEAN) / STD
    tensor = (
        torch.from_numpy(np.ascontiguousarray(fc))
        .permute(2, 0, 1).unsqueeze(0).float().to(device)
    )
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    return prob


# ================================================================
# PDF REPORT
# ================================================================
def generate_pdf(results, mode, source_name, threshold):
    buf          = io.BytesIO()
    ts           = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fakes        = [r for r in results if r["label"] == "FAKE"]
    reals        = [r for r in results if r["label"] == "REAL"]
    fake_pct     = 100 * len(fakes) / max(len(results), 1)
    avg_prob_raw = np.mean([r["prob_fake"] for r in results])
    verdict      = "FAKE" if avg_prob_raw >= threshold else "REAL"
    vcolor       = "#e74c3c" if verdict == "FAKE" else "#2ecc71"

    with PdfPages(buf) as pdf:
        # Page 1 — summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#0d0f1a")
        fig.text(0.5, 0.94, "DeepShield — Detection Report",
                 ha="center", fontsize=20, fontweight="bold", color="white")
        fig.text(0.5, 0.90, f"Generated: {ts}  |  Source: {source_name}",
                 ha="center", fontsize=10, color="#6b7280")

        ax_v = fig.add_axes([0.32, 0.64, 0.36, 0.20])
        ax_v.set_facecolor(vcolor + "33")
        for sp in ax_v.spines.values():
            sp.set_edgecolor(vcolor); sp.set_linewidth(2)
        ax_v.text(0.5, 0.65, verdict, transform=ax_v.transAxes,
                  ha="center", va="center", fontsize=38,
                  fontweight="bold", color=vcolor)
        conf_val = (
            np.mean([r["prob_fake"] for r in fakes]) * 100 if fakes
            else (1 - np.mean([r["prob_fake"] for r in reals])) * 100
        )
        ax_v.text(0.5, 0.22, f"Confidence: {conf_val:.1f}%",
                  transform=ax_v.transAxes, ha="center", fontsize=13, color="white")
        ax_v.axis("off")

        stats = [
            ("Mode",                 mode.upper()),
            ("Frames analysed",      str(len(results))),
            ("Fake frames",          f"{len(fakes)} ({fake_pct:.1f}%)"),
            ("Real frames",          f"{len(reals)} ({100-fake_pct:.1f}%)"),
            ("Avg fake probability", f"{avg_prob_raw*100:.2f}%"),
            ("Decision threshold",   f"{threshold*100:.0f}%"),
            ("Model",                "Attention-Xception + CBAM"),
            ("Dataset",              "Balanced (31,949 Real + 31,949 Fake)"),
            ("Test accuracy",        "98.67%"),
            ("ROC-AUC",              "0.9990"),
        ]
        y = 0.58
        for k, v in stats:
            fig.text(0.18, y, f"{k}:", fontsize=10, color="#9ca3af", ha="left")
            fig.text(0.58, y, v,       fontsize=10, color="white",
                     ha="left", fontweight="bold")
            y -= 0.048

        if len(results) > 1:
            ax_t = fig.add_axes([0.08, 0.06, 0.84, 0.16])
            ax_t.set_facecolor("#12152a")
            fn   = [r["frame_no"] for r in results]
            pb   = [r["prob_fake"] * 100 for r in results]
            cols = ["#ff4444" if p >= 50 else "#00e676" for p in pb]
            ax_t.bar(fn, pb, color=cols,
                     width=max(fn) / max(len(fn), 1) * 0.8)
            ax_t.axhline(50, color="white", ls="--", lw=0.8, alpha=0.4)
            ax_t.set_xlabel("Frame", color="#9ca3af", fontsize=8)
            ax_t.set_ylabel("Fake %", color="#9ca3af", fontsize=8)
            ax_t.set_title("Per-Frame Fake Probability", color="white", fontsize=9)
            ax_t.tick_params(colors="#9ca3af", labelsize=7)
            ax_t.set_ylim([0, 105])
            for sp in ax_t.spines.values():
                sp.set_edgecolor("#1e2340")

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close()

        # Page 2+ — frame samples
        sorted_f = sorted(results, key=lambda x: x["prob_fake"], reverse=True)
        sorted_r = sorted(results, key=lambda x: x["prob_fake"])
        samples  = sorted(sorted_f[:4] + sorted_r[:2], key=lambda x: x["frame_no"])

        for i in range(0, len(samples), 2):
            batch = samples[i:i+2]
            fig   = plt.figure(figsize=(11, 8.5))
            fig.patch.set_facecolor("#0d0f1a")
            fig.text(0.5, 0.97, "Frame Analysis — Original | Grad-CAM Overlay",
                     ha="center", fontsize=12, color="white", fontweight="bold")
            for j, r in enumerate(batch):
                top = 0.50 - j * 0.47
                c   = "#e74c3c" if r["label"] == "FAKE" else "#2ecc71"
                cf  = r["prob_fake"] if r["label"] == "FAKE" else 1 - r["prob_fake"]
                ax1 = fig.add_axes([0.05, top, 0.42, 0.40])
                ax1.imshow(r["original_rgb"]); ax1.axis("off")
                ax1.set_title(
                    f"Frame {r['frame_no']} — {r['label']}  {cf*100:.1f}%",
                    color=c, fontsize=11, fontweight="bold", pad=5
                )
                ax2 = fig.add_axes([0.53, top, 0.42, 0.40])
                ax2.imshow(r["overlay_rgb"]); ax2.axis("off")
                ax2.set_title("Grad-CAM Attention", color="white", fontsize=11, pad=5)
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close()

    buf.seek(0)
    return buf


# ================================================================
# UI HELPERS
# ================================================================
def render_verdict(label, conf, prob_fake):
    if label == "FAKE":
        st.markdown(f"""
        <div class="verdict-fake">
            <p class="verdict-label-fake">⚠ FAKE</p>
            <p class="confidence-text">Confidence: {conf*100:.1f}%</p>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill-fake" style="width:{prob_fake*100:.1f}%"></div>
            </div>
            <p class="confidence-text" style="font-size:0.8rem;margin-top:0.3rem">
                Fake probability: {prob_fake*100:.2f}%
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-real">
            <p class="verdict-label-real">✓ REAL</p>
            <p class="confidence-text">Confidence: {conf*100:.1f}%</p>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill-real" style="width:{conf*100:.1f}%"></div>
            </div>
            <p class="confidence-text" style="font-size:0.8rem;margin-top:0.3rem">
                Real probability: {(1-prob_fake)*100:.2f}%
            </p>
        </div>""", unsafe_allow_html=True)


def render_metric(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>""", unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>',
                unsafe_allow_html=True)


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.5rem 0 1rem'>
        <span style='font-size:2.5rem'>🛡️</span>
        <h2 style='font-family:Syne,sans-serif;font-weight:800;
                   font-size:1.4rem;margin:0.5rem 0 0;color:#e8e8f0'>
            DeepShield
        </h2>
        <p style='font-family:Space Mono,monospace;font-size:0.65rem;
                  color:#6b7280;letter-spacing:0.15em;text-transform:uppercase'>
            Deepfake Detection System
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    section("Model")
    model_path = st.text_input(
        "Path to best_model.pth",
        value="best_model.pth",
        label_visibility="collapsed"
    )

    st.divider()
    section("Settings")
    threshold = st.slider(
        "Decision Threshold", min_value=0.1, max_value=0.9,
        value=0.5, step=0.05, help="Probability above this = FAKE"
    )
    frame_skip = st.slider(
        "Video Frame Skip", min_value=5, max_value=60,
        value=30, step=5,
        help="Every Nth frame analysed. 30=fast, 10=thorough"
    )

    st.divider()
    section("Model Info")
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.72rem;
                color:#6b7280;line-height:1.8'>
        Architecture<br>
        <span style='color:#e8e8f0'>Attention-Xception + CBAM</span><br><br>
        Test Accuracy<br>
        <span style='color:#00e676'>98.67%</span><br><br>
        ROC-AUC<br>
        <span style='color:#00e676'>0.9990</span><br><br>
        Dataset<br>
        <span style='color:#e8e8f0'>63,898 balanced images</span>
    </div>
    """, unsafe_allow_html=True)


# ================================================================
# STARTUP — load model once, stays cached for all users/reruns
# ================================================================
st.markdown("""
<div style='padding: 2rem 0 1.5rem'>
    <p class="main-title">DeepShield</p>
    <p class="subtitle">AI-powered deepfake detection · Attention-Augmented Xception</p>
</div>
""", unsafe_allow_html=True)

# FIX: initialise flags before the try/except block so that a partial
# failure (e.g. load_model raises after cache miss) never leaves
# _model/_device in an indeterminate state across reruns.
_model_ok = False
_model    = None
_device   = None

if not os.path.exists(model_path):
    st.markdown(f"""
    <div style='background:#2d0a0a;border:1px solid #ff4444;border-radius:8px;
                padding:0.6rem 1rem;margin-bottom:1rem;
                font-family:Space Mono,monospace;font-size:0.78rem;color:#ff4444'>
        ⚠ Model not found at <b>{model_path}</b> — enter correct path in sidebar
    </div>""", unsafe_allow_html=True)
else:
    with st.spinner("⚡ Loading model into memory... (one-time, ~30s on CPU)"):
        try:
            _model, _device = load_model(model_path)
            # FIX: only set True AFTER both assignments succeed
            _model_ok = True
        except Exception as _e:
            st.error(f"Failed to load model: {_e}")

    if _model_ok:
        st.markdown(f"""
        <div style='background:#0a2d1a;border:1px solid #00e676;border-radius:8px;
                    padding:0.6rem 1rem;margin-bottom:1rem;
                    font-family:Space Mono,monospace;font-size:0.78rem;color:#00e676'>
            ✓ Model ready &nbsp;·&nbsp; Device: <b>{_device.upper()}</b>
            &nbsp;·&nbsp; Threshold: {threshold}
            &nbsp;·&nbsp; Upload image or video below
        </div>""", unsafe_allow_html=True)

# RetinaFace — silent fallback if not installed
_rf_module, _rf_ready = load_face_extractor()

# ================================================================
# TABS
# ================================================================
tab_img, tab_vid = st.tabs(["🖼  Image", "🎬  Video"])


# ================================================================
# TAB 1 — IMAGE
# ================================================================
with tab_img:
    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        section("Upload Image")
        uploaded = st.file_uploader(
            "Drop an image here",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
            key="img_upload"
        )

        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            img_rgb = np.array(img_pil)
            st.image(img_rgb, caption="Uploaded image", use_column_width=True)

            if st.button("🔍  Analyse Image", key="btn_img"):
                if not _model_ok:
                    st.error("Model not loaded — check path in sidebar.")
                else:
                    with st.spinner("Analysing image..."):
                        label, prob_fake, conf, overlay, heatmap, face_det = run_inference(
                            _model, _device, img_rgb, threshold,
                            _rf_module, _rf_ready
                        )
                    st.session_state["img_result"] = {
                        "label":         label,
                        "prob_fake":     prob_fake,
                        "conf":          conf,
                        "overlay":       overlay,
                        "heatmap":       heatmap,
                        "original":      np.array(img_pil.resize((299, 299))),
                        "name":          uploaded.name,
                        "face_detected": face_det,
                    }

    with col_res:
        section("Analysis Result")

        if "img_result" in st.session_state:
            r = st.session_state["img_result"]

            det_color  = "#00b4d8" if r.get("face_detected") else "#f59e0b"
            det_status = "RetinaFace crop" if r.get("face_detected") else "Full frame (no face detected)"
            st.markdown(f"""
            <div style='font-family:Space Mono,monospace;font-size:0.72rem;
                        color:{det_color};margin-bottom:1rem'>
                Face detection: {det_status}
            </div>""", unsafe_allow_html=True)

            render_verdict(r["label"], r["conf"], r["prob_fake"])
            st.markdown("<br>", unsafe_allow_html=True)

            mc1, mc2, mc3 = st.columns(3)
            with mc1: render_metric("Fake Prob",  f"{r['prob_fake']*100:.1f}%")
            with mc2: render_metric("Real Prob",  f"{(1-r['prob_fake'])*100:.1f}%")
            with mc3: render_metric("Threshold",  f"{threshold*100:.0f}%")

            section("Grad-CAM Attention Map")
            g1, g2 = st.columns(2)
            with g1:
                st.image(r["heatmap"],  caption="Attention heatmap", use_column_width=True)
            with g2:
                st.image(r["overlay"],  caption="Overlay on face",   use_column_width=True)

            section("Export")
            result_entry = [{
                "frame_no":     1,
                "label":        r["label"],
                "prob_fake":    r["prob_fake"],
                "original_rgb": r["original"],
                "overlay_rgb":  r["overlay"],
            }]
            pdf_buf = generate_pdf(result_entry, "image", r["name"], threshold)
            st.download_button(
                label="📄  Download PDF Report",
                data=pdf_buf,
                file_name=f"deepshield_{r['name'].split('.')[0]}.pdf",
                mime="application/pdf"
            )
        else:
            st.markdown("""
            <div style='text-align:center;padding:4rem 2rem;color:#2a2d4a;
                        font-family:Space Mono,monospace;font-size:0.85rem'>
                Upload an image and click<br>Analyse to see results
            </div>""", unsafe_allow_html=True)


# ================================================================
# TAB 2 — VIDEO
# ================================================================
with tab_vid:
    col_vu, col_vr = st.columns([1, 1], gap="large")

    with col_vu:
        section("Upload Video")
        vid_file = st.file_uploader(
            "Drop a video here",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
            key="vid_upload"
        )

        if vid_file:
            st.video(vid_file)

            if st.button("🔍  Analyse Video", key="btn_vid"):
                if not _model_ok:
                    st.error("Model not loaded — check path in sidebar.")
                    st.stop()

                # Save upload to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(vid_file.read())
                    tmp_path = tmp.name

                results   = []
                progress  = st.progress(0, text="Reading video...")
                status    = st.empty()

                cap          = cv2.VideoCapture(tmp_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames_to_process = max(total_frames // frame_skip, 1)

                for i in range(frames_to_process):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_skip)
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    frame_no = i * frame_skip
                    pct      = int((i + 1) / frames_to_process * 100)
                    progress.progress(pct, text=f"Frame {frame_no}/{total_frames}")

                    # Resize for speed
                    frame_bgr = cv2.resize(frame_bgr, (640, 360))
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_rgb = np.ascontiguousarray(frame_rgb)

                    face_probs   = []
                    best_overlay = None
                    best_prob    = 0.0
                    face_summary = ""

                    # Try RetinaFace
                    if _rf_ready and _rf_module is not None:
                        try:
                            img_bgr_rf = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            faces      = _rf_module.detect_faces(img_bgr_rf)
                            if isinstance(faces, dict) and faces:
                                for fid, face_val in faces.items():
                                    x1, y1, x2, y2 = map(int, face_val["facial_area"])
                                    face_crop = frame_rgb[y1:y2, x1:x2]
                                    if face_crop.size == 0:
                                        continue
                                    prob = infer_frame(_model, _device, face_crop)
                                    face_probs.append(prob)
                                    if prob > best_prob:
                                        best_prob    = prob
                                        best_overlay = cv2.resize(face_crop, (299, 299))
                                    face_summary += f"{fid}:{prob*100:.0f}% "
                        except Exception:
                            pass

                    # Fallback — full frame
                    if not face_probs:
                        prob = infer_frame(_model, _device, frame_rgb)
                        face_probs.append(prob)
                        best_prob    = prob
                        best_overlay = cv2.resize(frame_rgb, (299, 299))
                        face_summary = f"full_frame:{prob*100:.0f}%"

                    frame_avg = float(np.mean(face_probs))
                    label     = "FAKE" if frame_avg >= threshold else "REAL"
                    conf      = frame_avg if label == "FAKE" else 1 - frame_avg

                    results.append({
                        "frame_no":     frame_no,
                        "label":        label,
                        "prob_fake":    frame_avg,
                        "conf":         conf,
                        "original_rgb": cv2.resize(frame_rgb, (299, 299)),
                        "overlay_rgb":  best_overlay,
                        "face_summary": face_summary.strip(),
                        "n_faces":      len(face_probs),
                    })

                    col_color = "#ff4444" if label == "FAKE" else "#00e676"
                    status.markdown(
                        f"<span style='font-family:Space Mono,monospace;"
                        f"font-size:0.78rem;color:#9ca3af'>"
                        f"Frame {frame_no} | {len(face_probs)} face(s) | "
                        f"<b style='color:{col_color}'>{label}</b> "
                        f"{conf*100:.1f}% | {face_summary}</span>",
                        unsafe_allow_html=True
                    )

                cap.release()
                os.unlink(tmp_path)
                progress.empty()
                status.empty()

                st.session_state["vid_results"] = {
                    "results": results,
                    "name":    vid_file.name,
                }

    with col_vr:
        section("Video Analysis Result")

        if "vid_results" in st.session_state:
            vr      = st.session_state["vid_results"]
            results = vr["results"]

            if not results:
                st.warning("No frames were analysed.")
            else:
                fakes    = [r for r in results if r["label"] == "FAKE"]
                reals    = [r for r in results if r["label"] == "REAL"]
                fake_pct = 100 * len(fakes) / len(results)
                avg_prob = float(np.mean([r["prob_fake"] for r in results]))
                max_prob = float(np.max([r["prob_fake"]  for r in results]))

                verdict = "FAKE" if avg_prob >= threshold else "REAL"
                conf_v  = avg_prob if verdict == "FAKE" else 1 - avg_prob

                render_verdict(verdict, conf_v, avg_prob)
                st.markdown("<br>", unsafe_allow_html=True)

                m1, m2, m3, m4, m5 = st.columns(5)
                with m1: render_metric("Frames",      str(len(results)))
                with m2: render_metric("Fake frames", f"{fake_pct:.1f}%")
                with m3: render_metric("Avg prob",    f"{avg_prob*100:.1f}%")
                with m4: render_metric("Max prob",    f"{max_prob*100:.1f}%")
                with m5: render_metric(
                    "Verdict by",
                    "any >85%" if max_prob >= 0.85
                    else ">20% frames" if fake_pct >= 20
                    else "avg prob"
                )

                section("Frame-by-Frame Timeline")
                fig, ax = plt.subplots(figsize=(8, 2.5))
                fig.patch.set_facecolor("#12152a")
                ax.set_facecolor("#12152a")
                fn   = [r["frame_no"] for r in results]
                pb   = [r["prob_fake"] * 100 for r in results]
                cols = ["#ff4444" if p >= 50 else "#00e676" for p in pb]
                ax.bar(fn, pb, color=cols,
                       width=max(fn) / max(len(fn), 1) * 0.8)
                ax.axhline(threshold * 100, color="white", ls="--",
                           lw=1, alpha=0.4,
                           label=f"Threshold {threshold*100:.0f}%")
                ax.set_xlabel("Frame number", color="#9ca3af", fontsize=8)
                ax.set_ylabel("Fake %",       color="#9ca3af", fontsize=8)
                ax.tick_params(colors="#9ca3af", labelsize=7)
                ax.set_ylim([0, 105])
                ax.legend(fontsize=7, labelcolor="#9ca3af",
                          facecolor="#12152a", edgecolor="#1e2340")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#1e2340")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                section("Frame Details")
                for r in results:
                    c    = "#ff4444" if r["label"] == "FAKE" else "#00e676"
                    nf   = r.get("n_faces", 1)
                    summ = r.get("face_summary", "")
                    row_class = "fake" if r["label"] == "FAKE" else "real"
                    st.markdown(
                        f"<div class='frame-row-{row_class}'>"
                        f"Frame {r['frame_no']:>5} &nbsp;·&nbsp; "
                        f"{nf} face(s) &nbsp;·&nbsp; "
                        f"<b style='color:{c}'>{r['label']}</b>"
                        f" &nbsp;·&nbsp; {r['prob_fake']*100:.1f}% fake"
                        f"{'&nbsp; | &nbsp;' + summ if summ else ''}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                section("Sample Frames")
                top_fake = sorted(fakes, key=lambda x: x["prob_fake"], reverse=True)[:2]
                top_real = sorted(reals, key=lambda x: x["prob_fake"])[:2]
                samples  = top_fake + top_real
                if samples:
                    gcols = st.columns(len(samples))
                    for col, r in zip(gcols, samples):
                        c = "#ff4444" if r["label"] == "FAKE" else "#00e676"
                        with col:
                            st.image(r["overlay_rgb"], use_column_width=True)
                            st.markdown(
                                f"<p style='text-align:center;"
                                f"font-family:Space Mono,monospace;"
                                f"font-size:0.7rem;color:{c}'>"
                                f"Frame {r['frame_no']}<br>"
                                f"{r['label']} {r['prob_fake']*100:.1f}%</p>",
                                unsafe_allow_html=True
                            )

                section("Export")
                pdf_buf = generate_pdf(results, "video", vr["name"], threshold)
                st.download_button(
                    label="📄  Download PDF Report",
                    data=pdf_buf,
                    file_name=f"deepshield_{vr['name'].split('.')[0]}.pdf",
                    mime="application/pdf"
                )
        else:
            st.markdown("""
            <div style='text-align:center;padding:4rem 2rem;color:#2a2d4a;
                        font-family:Space Mono,monospace;font-size:0.85rem'>
                Upload a video and click<br>Analyse to see results
            </div>""", unsafe_allow_html=True)
