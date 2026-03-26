import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="Kisan AI – Smart Farming",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;500;600;700;800&family=Hind:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #fdf6ec !important;
    font-family: 'Hind', sans-serif !important;
    color: #2d1e0f !important;
}
[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="stSidebar"],[data-testid="stDecoration"] { display: none !important; }

.block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; margin: 0 !important; }
[data-testid="stVerticalBlock"]              { gap: 0 !important; padding: 0 !important; }
[data-testid="stVerticalBlockBorderWrapper"] { padding: 0 !important; }
div[data-testid="element-container"]         { padding: 0 !important; margin: 0 !important; }
[data-testid="stHorizontalBlock"]            { gap: 24px !important; align-items: stretch !important; padding: 0 !important; }
[data-testid="column"]                       { padding: 0 !important; min-width: 0 !important; }

/* ══════════ BANNER ══════════ */
.top-banner {
    background: linear-gradient(135deg, #c97508 0%, #f5c030 52%, #e07810 100%);
    padding: 32px 72px 30px;
    position: relative; overflow: hidden;
    display: flex; align-items: center; justify-content: space-between;
}
.banner-bg-text {
    position: absolute; right: 90px; top: -44px;
    font-size: 260px; font-family: 'Baloo 2', cursive; font-weight: 800;
    color: rgba(255,255,255,0.07); line-height: 1; pointer-events: none;
}
.banner-tag {
    display: inline-block;
    background: rgba(255,255,255,0.22); border: 1.5px solid rgba(255,255,255,0.45);
    border-radius: 30px; padding: 3px 14px;
    font-size: 12px; font-weight: 600; color: #fff; letter-spacing: 0.06em; margin-bottom: 10px;
}
.banner-title {
    font-family: 'Baloo 2', cursive; font-weight: 800; font-size: 56px;
    color: #fff; line-height: 1.0; text-shadow: 0 2px 12px rgba(0,0,0,0.15); margin-bottom: 6px;
}
.banner-title span { color: #3d1f00; }
.banner-subtitle { font-size: 15px; color: rgba(255,255,255,0.9); line-height: 1.55; }
.banner-icon {
    font-size: 88px; line-height: 1;
    filter: drop-shadow(0 4px 12px rgba(0,0,0,0.15));
    animation: float 3s ease-in-out infinite;
}
@keyframes float { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-10px);} }

/* ══════════ STEPS ══════════ */
.step-guide {
    background: #fff8ee; border-bottom: 2px solid #f0d090;
    padding: 13px 72px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
}
.step-pill {
    display: flex; align-items: center; gap: 7px;
    background: #fff; border: 1.5px solid #f0c870; border-radius: 30px; padding: 5px 16px;
    font-family: 'Baloo 2', cursive; font-size: 14px; font-weight: 600; color: #7a4800; white-space: nowrap;
}
.step-num {
    width: 22px; height: 22px; background: #e8a020; border-radius: 50%;
    color: #fff; font-size: 12px; font-weight: 700;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.step-arrow { color: #cba040; font-size: 20px; font-weight: 700; }

/* ══════════ OUTER PADDING ══════════ */
.outer-pad { padding: 28px 72px 32px; background: #fdf6ec; }

/* ══════════════════════════════════════════════════
   CARD BORDER SEGMENTS
   top / mid / bot split so Streamlit widgets can
   be injected between html blocks safely
══════════════════════════════════════════════════ */
.seg-top {
    background: #fff;
    border: 2px solid #f0d090;
    border-radius: 20px 20px 0 0;
    /* no bottom border — seamlessly continues into next seg */
    border-bottom: none;
}
.seg-mid {
    background: #fff;
    border-left:  2px solid #f0d090;
    border-right: 2px solid #f0d090;
}
.seg-bot {
    background: #fff;
    border: 2px solid #f0d090;
    border-top: none;
    border-radius: 0 0 20px 20px;
    box-shadow: 0 6px 28px rgba(200,130,0,0.10);
}

/* card header */
.card-header {
    padding: 18px 24px 16px;
    border-bottom: 2px dashed #f0d898;
    display: flex; align-items: center; gap: 12px;
}
.card-emoji  { font-size: 32px; line-height: 1; flex-shrink: 0; }
.card-title  { font-family: 'Baloo 2', cursive; font-size: 21px; font-weight: 700; color: #2d1e0f; line-height: 1.1; }
.card-sub    { font-size: 12px; color: #a07040; margin-top: 3px; }

/* ══════════════════════════════════════════════════
   FIELD ROW
   Structure per field:
     <div class="field-label seg-mid">  label + hint  </div>
     <div class="field-input seg-mid">  input widget  </div>

   .field-label  — generous top padding so text sits
                   well below the card segment's top edge
   .field-input  — bottom padding creates gap before
                   next label block starts
══════════════════════════════════════════════════ */
.field-label {
    /* top:20 pushes the label DOWN inside the white segment so
       it is fully visible and never clipped by the border above */
    padding: 20px 24px 4px 24px;
}

.f-label {
    font-family: 'Baloo 2', cursive;
    font-size: 17px; font-weight: 700; color: #5a3800;
    display: block; line-height: 1.25; margin-bottom: 4px;
}
.f-hint {
    font-size: 12px; color: #b08040;
    display: block; line-height: 1.4;
}

.field-input {
    /* bottom:18 gives breathing room between this input
       and the label block of the NEXT field */
    padding: 8px 24px 18px 24px;
}

/* last field in a card — less bottom so bot-seg cap looks right */
.field-input.last { padding-bottom: 10px; }

/* ══════════ NUMBER INPUT ══════════ */
div[data-testid="stNumberInput"] { margin: 0 !important; padding: 0 !important; }
div[data-testid="stNumberInput"] > label { display: none !important; }
div[data-testid="stNumberInput"] > div   { height: 50px !important; min-height: 50px !important; }

div[data-testid="stNumberInput"] input {
    background: #fffbf4 !important;
    border: 2px solid #e8c870 !important; border-radius: 12px !important;
    color: #2d1e0f !important;
    font-family: 'Baloo 2', cursive !important; font-size: 18px !important; font-weight: 600 !important;
    padding: 9px 14px !important; height: 50px !important; width: 100% !important;
    transition: border-color 0.22s, box-shadow 0.22s !important;
}
div[data-testid="stNumberInput"] input::placeholder {
    color: #c8a060 !important; font-weight: 400 !important;
    font-size: 14px !important; font-style: italic !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #e8851a !important; background: #fff9f0 !important;
    box-shadow: 0 0 0 3px rgba(232,133,26,0.14) !important; outline: none !important;
}
div[data-testid="stNumberInput"] input:hover { border-color: #d4a030 !important; }
div[data-testid="stNumberInput"] button {
    background: transparent !important; border: none !important;
    color: #c8900a !important; font-size: 15px !important;
    transition: color 0.2s !important; padding: 0 5px !important;
}
div[data-testid="stNumberInput"] button:hover {
    color: #e8601a !important; background: rgba(232,160,30,0.1) !important;
}

/* ══════════ BUTTON ══════════ */
.stButton { margin: 0 !important; padding: 0 !important; }
.btn-area { padding: 18px 24px 24px 24px; }

.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #e8851a 0%, #f5c020 100%) !important;
    color: #fff !important; border: none !important; border-radius: 13px !important;
    font-family: 'Baloo 2', cursive !important; font-size: 21px !important;
    font-weight: 800 !important; letter-spacing: 0.02em !important;
    padding: 15px 20px !important; height: auto !important;
    box-shadow: 0 6px 22px rgba(232,133,26,0.40) !important;
    text-shadow: 0 1px 4px rgba(0,0,0,0.18) !important;
    transition: all 0.25s ease !important; cursor: pointer !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #f09020 0%, #fad030 100%) !important;
    box-shadow: 0 12px 36px rgba(232,133,26,0.55) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ══════════ WARN ══════════ */
.warn-box {
    background: #fff5e0; border: 2px solid #f0b040; border-radius: 14px;
    padding: 14px 24px; display: flex; align-items: flex-start; gap: 12px;
    margin: 0 72px 4px;
    font-family: 'Baloo 2', cursive; font-size: 17px; font-weight: 600; color: #7a4800;
    animation: shake 0.4s ease; line-height: 1.5;
}
@keyframes shake {
    0%,100%{transform:translateX(0);}
    20%,60%{transform:translateX(-6px);}
    40%,80%{transform:translateX(6px);}
}

/* ══════════ RESULTS ══════════ */
.results-wrap { padding: 10px 72px 32px; animation: up 0.5s ease both; }
@keyframes up { from{opacity:0;transform:translateY(22px);} to{opacity:1;transform:translateY(0);} }

.res-banner {
    background: linear-gradient(90deg,#2e7d1a,#52a830); border-radius: 16px;
    padding: 16px 32px; display: flex; align-items: center; gap: 16px;
    margin-bottom: 18px; box-shadow: 0 4px 20px rgba(46,125,26,0.25);
}
.res-banner-icon  { font-size: 30px; }
.res-banner-title { font-family:'Baloo 2',cursive; font-size:22px; font-weight:700; color:#fff; }
.res-banner-sub   { font-size:14px; color:rgba(255,255,255,0.8); }

.res-cards { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; }
.res-card {
    background:#fff; border-radius:18px; border:2.5px solid;
    box-shadow:0 4px 24px rgba(0,0,0,0.07); overflow:hidden;
    animation: pop 0.4s cubic-bezier(0.34,1.56,0.64,1) both;
}
.res-card:nth-child(1){ border-color:#52a830; animation-delay:.05s; }
.res-card:nth-child(2){ border-color:#c8800a; animation-delay:.12s; }
.res-card:nth-child(3){ border-color:#1a80c8; animation-delay:.19s; }
@keyframes pop { from{opacity:0;transform:scale(.88);} to{opacity:1;transform:scale(1);} }

.res-card-top { padding:16px 22px 12px; display:flex; align-items:center; gap:12px; }
.res-card:nth-child(1) .res-card-top { background:#f0fbe8; }
.res-card:nth-child(2) .res-card-top { background:#fff5e8; }
.res-card:nth-child(3) .res-card-top { background:#e8f4ff; }
.res-card-icon  { font-size:28px; }
.res-card-label { font-family:'Baloo 2',cursive; font-size:13px; font-weight:700; color:#5a3800; text-transform:uppercase; letter-spacing:.06em; }
.res-card-body  { padding:16px 22px 20px; }
.res-value { font-family:'Baloo 2',cursive; font-size:34px; font-weight:800; line-height:1.15; margin-bottom:6px; word-break:break-word; }
.res-card:nth-child(1) .res-value { color:#2e7d1a; }
.res-card:nth-child(2) .res-value { color:#a05a00; }
.res-card:nth-child(3) .res-value { color:#0a5a9e; }
.res-desc { font-size:14px; color:#907050; line-height:1.5; }

/* ══════════ FOOTER ══════════ */
.footer {
    background:#2d1e0f; padding:14px 72px;
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;
}
.footer-brand { font-family:'Baloo 2',cursive; font-size:17px; font-weight:700; color:#f5c040; }
.footer-text  { font-family:'Baloo 2',cursive; font-size:13px; color:rgba(255,230,180,.45); }

@media(max-width:960px){
    .top-banner,.step-guide,.outer-pad,.results-wrap,.warn-box,.footer{padding-left:18px!important;padding-right:18px!important;}
    .res-cards{grid-template-columns:1fr;}
    .banner-icon{display:none;}
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_models():
    crop_model = joblib.load(os.path.join(model_dir, "crop_model.pkl"))
    soil_model = joblib.load(os.path.join(model_dir, "soil_model.pkl"))
    fert_model  = joblib.load(os.path.join(model_dir, "fertilizer_model.pkl"))
    return crop_model, soil_model, fert_model

crop_model, soil_model, fert_model = load_models()

# ── Banner + Steps ────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
    <div class="banner-bg-text">K</div>
    <div>
        <div class="banner-tag">AI Farming Assistant</div>
        <div class="banner-title">Kisan <span>AI</span></div>
        <div class="banner-subtitle">
            Enter your soil and weather details below.<br>
            Get the best crop, soil type &amp; fertilizer recommendation instantly.
        </div>
    </div>
    <div class="banner-icon">🌾</div>
</div>
<div class="step-guide">
    <div class="step-pill"><div class="step-num">1</div> Fill Soil Nutrients</div>
    <div class="step-arrow">›</div>
    <div class="step-pill"><div class="step-num">2</div> Fill Weather Info</div>
    <div class="step-arrow">›</div>
    <div class="step-pill"><div class="step-num">3</div> Fill Soil Properties</div>
    <div class="step-arrow">›</div>
    <div class="step-pill"><div class="step-num">4</div> Click Get Results</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="outer-pad">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3, gap="medium")

# ─────────────────────────────────────────────────────────
# CARD 1 — Soil Nutrients
# Each field = seg-mid label div  +  seg-mid input div
# Card opens with seg-top (header + first label)
# Card closes with seg-bot (empty cap with box-shadow)
# ─────────────────────────────────────────────────────────
with col1:

    # TOP: card header
    st.markdown("""
    <div class="seg-top">
        <div class="card-header">
            <div class="card-emoji">🧪</div>
            <div>
                <div class="card-title">Soil Nutrients</div>
                <div class="card-sub">From your soil test report (kg/ha)</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # FIELD: Nitrogen
    st.markdown("""<div class="seg-mid"><div class="field-label">
        <span class="f-label">Nitrogen (N)</span>
        <span class="f-hint">Typical range: 0 – 140 kg/ha</span>
    </div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="seg-mid"><div class="field-input">', unsafe_allow_html=True)
    N = st.number_input("Nitrogen", min_value=0.0, max_value=500.0, value=None,
                        placeholder="e.g. 50", label_visibility="collapsed", key="N")
    st.markdown('</div></div>', unsafe_allow_html=True)

    # FIELD: Phosphorus
    st.markdown("""<div class="seg-mid"><div class="field-label">
        <span class="f-label">Phosphorus (P)</span>
        <span class="f-hint">Typical range: 0 – 145 kg/ha</span>
    </div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="seg-mid"><div class="field-input">', unsafe_allow_html=True)
    P = st.number_input("Phosphorus", min_value=0.0, max_value=500.0, value=None,
                        placeholder="e.g. 40", label_visibility="collapsed", key="P")
    st.markdown('</div></div>', unsafe_allow_html=True)

    # FIELD: Potassium
    st.markdown("""<div class="seg-mid"><div class="field-label">
        <span class="f-label">Potassium (K)</span>
        <span class="f-hint">Typical range: 0 – 205 kg/ha</span>
    </div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="seg-mid"><div class="field-input last">', unsafe_allow_html=True)
    K = st.number_input("Potassium", min_value=0.0, max_value=500.0, value=None,
                        placeholder="e.g. 35", label_visibility="collapsed", key="K")
    st.markdown('</div></div>', unsafe_allow_html=True)

    # BOTTOM CAP
    st.markdown('<div class="seg-bot" style="height:18px;"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# CARD 2 — Weather & Water
# ─────────────────────────────────────────────────────────
with col2:

    st.markdown("""
    <div class="seg-top">
        <div class="card-header">
            <div class="card-emoji">🌤️</div>
            <div>
                <div class="card-title">Weather &amp; Water</div>
                <div class="card-sub">Local climate conditions</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Temperature
    st.markdown("""<div class="seg-mid"><div class="field-label">
        <span class="f-label">Temperature (°C)</span>
        <span class="f-hint">Average daytime temperature</span>
    </div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="seg-mid"><div class="field-input">', unsafe_allow_html=True)
    temp = st.number_input("Temperature", min_value=-20.0, max_value=60.0, value=None,
                           placeholder="e.g. 28", label_visibility="collapsed", key="temp")
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Humidity
    st.markdown("""<div class="seg-mid"><div class="field-label">
        <span class="f-label">Humidity (%)</span>
        <span class="f-hint">Relative humidity in the air</span>
    </div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="seg-mid"><div class="field-input">', unsafe_allow_html=True)
    humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=None,
                               placeholder="e.g. 65", label_visibility="collapsed", key="humidity")
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Rainfall
    st.markdown("""<div class="seg-mid"><div class="field-label">
        <span class="f-label">Rainfall (mm)</span>
        <span class="f-hint">Annual or seasonal rainfall</span>
    </div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="seg-mid"><div class="field-input">', unsafe_allow_html=True)
    rainfall = st.number_input("Rainfall", min_value=0.0, value=None,
                               placeholder="e.g. 200", label_visibility="collapsed", key="rainfall")
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Soil Moisture
    st.markdown("""<div class="seg-mid"><div class="field-label">
        <span class="f-label">Soil Moisture (%)</span>
        <span class="f-hint">Current moisture in topsoil</span>
    </div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="seg-mid"><div class="field-input last">', unsafe_allow_html=True)
    moisture = st.number_input("Moisture", min_value=0.0, max_value=100.0, value=None,
                               placeholder="e.g. 50", label_visibility="collapsed", key="moisture")
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="seg-bot" style="height:18px;"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# CARD 3 — Soil Properties
# ─────────────────────────────────────────────────────────
with col3:

    st.markdown("""
    <div class="seg-top">
        <div class="card-header">
            <div class="card-emoji">🌍</div>
            <div>
                <div class="card-title">Soil Properties</div>
                <div class="card-sub">Chemical quality of your soil</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Soil pH
    st.markdown("""<div class="seg-mid"><div class="field-label">
        <span class="f-label">Soil pH</span>
        <span class="f-hint">0 = very acidic · 7 = neutral · 14 = alkaline</span>
    </div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="seg-mid"><div class="field-input">', unsafe_allow_html=True)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=None,
                         placeholder="e.g. 6.5", label_visibility="collapsed", key="ph")
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Organic Carbon
    st.markdown("""<div class="seg-mid"><div class="field-label">
        <span class="f-label">Organic Carbon (%)</span>
        <span class="f-hint">Organic matter content in soil</span>
    </div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="seg-mid"><div class="field-input">', unsafe_allow_html=True)
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=None,
                                     placeholder="e.g. 0.8", label_visibility="collapsed", key="oc")
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Electrical Conductivity
    st.markdown("""<div class="seg-mid"><div class="field-label">
        <span class="f-label">Electrical Conductivity (dS/m)</span>
        <span class="f-hint">Soil salinity · typical range: 0 – 4</span>
    </div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="seg-mid"><div class="field-input last">', unsafe_allow_html=True)
    ec = st.number_input("Electrical Conductivity", min_value=0.0, value=None,
                         placeholder="e.g. 1.2", label_visibility="collapsed", key="ec")
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Button inside bottom cap
    st.markdown('<div class="seg-bot"><div class="btn-area">', unsafe_allow_html=True)
    predict_btn = st.button("🌱  Get Recommendation", use_container_width=True, key="predict")
    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close outer-pad

# ── Validation & Prediction ───────────────────────────────────────────────────
if predict_btn:
    fields = {
        "Nitrogen (N)": N, "Phosphorus (P)": P, "Potassium (K)": K,
        "Temperature": temp, "Humidity": humidity, "Rainfall": rainfall,
        "Soil Moisture": moisture, "Soil pH": ph,
        "Organic Carbon": organic_carbon, "Electrical Conductivity": ec,
    }
    missing = [k for k, v in fields.items() if v is None]

    if missing:
        st.markdown(f"""
        <div class="warn-box">
            <span style="font-size:26px;flex-shrink:0">⚠️</span>
            <span>Please fill in: <strong>{",  ".join(missing)}</strong></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing your soil data..."):
            crop = crop_model.predict([[N, P, K, temp, humidity, ph, rainfall]])[0]
            soil = soil_model.predict([[temp, humidity, moisture, ph, organic_carbon, ec, N, P, K]])[0]
            fert  = fert_model.predict([[temp, humidity, moisture, N, P, K]])[0]

        st.markdown(f"""
        <div class="results-wrap">
            <div class="res-banner">
                <div class="res-banner-icon">✅</div>
                <div>
                    <div class="res-banner-title">Analysis Complete — Your Recommendations</div>
                    <div class="res-banner-sub">Based on your soil and weather inputs</div>
                </div>
            </div>
            <div class="res-cards">
                <div class="res-card">
                    <div class="res-card-top">
                        <div class="res-card-icon">🌾</div>
                        <div class="res-card-label">Best Crop to Grow</div>
                    </div>
                    <div class="res-card-body">
                        <div class="res-value">{crop}</div>
                        <div class="res-desc">Best suited for your soil nutrients and local climate.</div>
                    </div>
                </div>
                <div class="res-card">
                    <div class="res-card-top">
                        <div class="res-card-icon">🏔️</div>
                        <div class="res-card-label">Soil Type Detected</div>
                    </div>
                    <div class="res-card-body">
                        <div class="res-value">{soil}</div>
                        <div class="res-desc">Classified from moisture, pH, carbon &amp; salinity.</div>
                    </div>
                </div>
                <div class="res-card">
                    <div class="res-card-top">
                        <div class="res-card-icon">🧴</div>
                        <div class="res-card-label">Fertilizer to Apply</div>
                    </div>
                    <div class="res-card-body">
                        <div class="res-value">{fert}</div>
                        <div class="res-desc">Corrects nutrient gaps and boosts field yield.</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-brand">Kisan AI</div>
    <div class="footer-text">Smart Farming · AI Soil Analysis System · v2.0</div>
</div>
""", unsafe_allow_html=True)
