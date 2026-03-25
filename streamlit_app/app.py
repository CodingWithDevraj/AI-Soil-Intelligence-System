import streamlit as st
import joblib
import numpy as np
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kisan AI - Smart Farming",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ───────────────────────────────────────────────────────────────────
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

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── Kill ALL Streamlit spacing ── */
.block-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}

section[data-testid="stMain"] > div {
    padding: 0 !important;
    margin: 0 !important;
}

/* Kill gap Streamlit adds between every st.markdown / st.columns block */
[data-testid="stVerticalBlock"] {
    gap: 0 !important;
    padding: 0 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    padding: 0 !important;
}

div[data-testid="element-container"] {
    padding: 0 !important;
    margin: 0 !important;
}

/* Column wrapper gap */
[data-testid="stHorizontalBlock"] {
    gap: 0 !important;
    padding: 0 !important;
}

/* ════════════════════════════════
   TOP BANNER
════════════════════════════════ */
.top-banner {
    background: linear-gradient(135deg, #d4820a 0%, #f5c030 50%, #e07810 100%);
    padding: 28px 52px 26px;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 130px;
}

.banner-bg-text {
    position: absolute;
    right: 120px; top: -40px;
    font-size: 260px;
    font-family: 'Baloo 2', cursive;
    font-weight: 800;
    color: rgba(255,255,255,0.07);
    line-height: 1;
    pointer-events: none;
}

.banner-tag {
    display: inline-block;
    background: rgba(255,255,255,0.22);
    border: 1.5px solid rgba(255,255,255,0.45);
    border-radius: 30px;
    padding: 3px 14px;
    font-size: 12px;
    font-weight: 600;
    color: #fff;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}

.banner-title {
    font-family: 'Baloo 2', cursive;
    font-weight: 800;
    font-size: 52px;
    color: #fff;
    line-height: 1.0;
    text-shadow: 0 2px 12px rgba(0,0,0,0.15);
    margin-bottom: 4px;
}
.banner-title span { color: #3d1f00; }

.banner-subtitle {
    font-size: 15px;
    color: rgba(255,255,255,0.88);
    font-weight: 400;
    line-height: 1.5;
}

.banner-icon {
    font-size: 88px;
    line-height: 1;
    filter: drop-shadow(0 4px 12px rgba(0,0,0,0.15));
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-10px); }
}

/* ════════════════════════════════
   STEP GUIDE
════════════════════════════════ */
.step-guide {
    background: #fff8ee;
    border-bottom: 2px solid #f0d090;
    padding: 12px 52px;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    /* no margin, no gap — flush to banner above and cards below */
}

.step-pill {
    display: flex;
    align-items: center;
    gap: 7px;
    background: #fff;
    border: 1.5px solid #f0c870;
    border-radius: 30px;
    padding: 5px 16px;
    font-family: 'Baloo 2', cursive;
    font-size: 15px;
    font-weight: 600;
    color: #7a4800;
}

.step-num {
    width: 22px; height: 22px;
    background: #e8a020;
    border-radius: 50%;
    color: #fff;
    font-size: 12px;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}

.step-arrow { color: #cba040; font-size: 20px; font-weight: 700; }

/* ════════════════════════════════
   FORM WRAPPER — directly follows step guide
════════════════════════════════ */
.form-wrapper {
    padding: 20px 40px 20px;
    background: #fdf6ec;
    /* No extra top margin — sits right under the step guide */
}

/* ── Cards ── */
.form-card {
    background: #fff;
    border-radius: 20px;
    border: 2px solid #f0d090;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(200,130,0,0.09);
    display: flex;
    flex-direction: column;
    height: 100%;
    transition: box-shadow 0.3s, transform 0.3s;
    animation: cardUp 0.45s ease both;
}

.form-card:nth-child(1) { animation-delay: 0.05s; }
.form-card:nth-child(2) { animation-delay: 0.12s; }
.form-card:nth-child(3) { animation-delay: 0.19s; }

.form-card:hover {
    box-shadow: 0 10px 40px rgba(200,130,0,0.18);
    transform: translateY(-3px);
}

@keyframes cardUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

.card-top {
    padding: 18px 28px 14px;
    border-bottom: 2px dashed #f5e0b0;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-shrink: 0;
    background: #fffdf8;
}

.card-emoji { font-size: 38px; line-height: 1; }

.card-title {
    font-family: 'Baloo 2', cursive;
    font-size: 24px;
    font-weight: 700;
    color: #2d1e0f;
    line-height: 1.1;
}

.card-hint { font-size: 13px; color: #a07040; margin-top: 1px; }

.card-body {
    padding: 16px 28px 22px;
    flex: 1;
}

/* ── Field Labels ── */
.field-label {
    font-family: 'Baloo 2', cursive;
    font-size: 18px;
    font-weight: 700;
    color: #5a3800;
    margin-top: 14px;
    margin-bottom: 2px;
    display: block;
    line-height: 1.2;
}
.field-label.first { margin-top: 0; }

.field-hint {
    font-size: 12px;
    color: #b08040;
    margin-bottom: 4px;
    display: block;
    line-height: 1.3;
}

/* ════════════════════════════════
   NUMBER INPUTS
════════════════════════════════ */
div[data-testid="stNumberInput"] {
    margin: 0 !important;
    padding: 0 !important;
}
div[data-testid="stNumberInput"] > label { display: none !important; }
div[data-testid="stNumberInput"] > div {
    height: 44px !important;
    min-height: 44px !important;
}

div[data-testid="stNumberInput"] input {
    background: #fffbf4 !important;
    border: 2px solid #e8c870 !important;
    border-radius: 10px !important;
    color: #2d1e0f !important;
    font-family: 'Baloo 2', cursive !important;
    font-size: 19px !important;
    font-weight: 600 !important;
    padding: 8px 14px !important;
    height: 44px !important;
    transition: border-color 0.22s, box-shadow 0.22s !important;
    width: 100% !important;
}

div[data-testid="stNumberInput"] input::placeholder {
    color: #c8a060 !important;
    font-weight: 400 !important;
    font-size: 15px !important;
    font-style: italic !important;
}

div[data-testid="stNumberInput"] input:focus {
    border-color: #e8851a !important;
    background: #fff9f0 !important;
    box-shadow: 0 0 0 3px rgba(232,133,26,0.13) !important;
    outline: none !important;
}
div[data-testid="stNumberInput"] input:hover { border-color: #d4a030 !important; }

div[data-testid="stNumberInput"] button {
    background: transparent !important;
    border: none !important;
    color: #c8900a !important;
    font-size: 16px !important;
    transition: color 0.2s !important;
    padding: 0 6px !important;
}
div[data-testid="stNumberInput"] button:hover {
    color: #e8601a !important;
    background: rgba(232,160,30,0.1) !important;
}

/* ════════════════════════════════
   PREDICT BUTTON
════════════════════════════════ */
.stButton { margin: 0 !important; padding: 0 !important; }
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #e8851a 0%, #f5c020 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Baloo 2', cursive !important;
    font-size: 22px !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em !important;
    padding: 15px 28px !important;
    height: auto !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 6px 22px rgba(232,133,26,0.38) !important;
    text-shadow: 0 1px 4px rgba(0,0,0,0.18) !important;
    margin-top: 20px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #f09020 0%, #fad030 100%) !important;
    box-shadow: 0 12px 36px rgba(232,133,26,0.52) !important;
    transform: translateY(-2px) scale(1.01) !important;
}
.stButton > button:active { transform: translateY(0) scale(0.99) !important; }

/* ════════════════════════════════
   COLUMN GAP INSIDE FORM
════════════════════════════════ */
.cols-wrapper [data-testid="stHorizontalBlock"] {
    gap: 20px !important;
    align-items: stretch !important;
}

.cols-wrapper [data-testid="column"] {
    padding: 0 !important;
}

/* ════════════════════════════════
   VALIDATION WARNING
════════════════════════════════ */
.warn-box {
    background: #fff5e0;
    border: 2px solid #f0b040;
    border-radius: 14px;
    padding: 14px 22px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 12px 40px 0;
    font-family: 'Baloo 2', cursive;
    font-size: 17px;
    font-weight: 600;
    color: #7a4800;
    animation: shake 0.4s ease;
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    20%, 60%  { transform: translateX(-6px); }
    40%, 80%  { transform: translateX(6px); }
}

/* ════════════════════════════════
   RESULTS
════════════════════════════════ */
.results-wrapper {
    padding: 18px 40px 28px;
    animation: resultsIn 0.5s ease both;
}

@keyframes resultsIn {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0); }
}

.results-banner {
    background: linear-gradient(90deg, #2e7d1a, #52a830);
    border-radius: 16px;
    padding: 16px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 18px;
    box-shadow: 0 4px 20px rgba(46,125,26,0.25);
}

.results-banner-icon { font-size: 30px; }
.results-banner-text {
    font-family: 'Baloo 2', cursive;
    font-size: 23px;
    font-weight: 700;
    color: #fff;
}
.results-banner-sub { font-size: 14px; color: rgba(255,255,255,0.8); }

.result-cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
}

.result-card {
    background: #fff;
    border-radius: 20px;
    border: 2.5px solid;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07);
    animation: cardPop 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}

.result-card:nth-child(1) { border-color: #52a830; animation-delay: 0.05s; }
.result-card:nth-child(2) { border-color: #c8800a; animation-delay: 0.12s; }
.result-card:nth-child(3) { border-color: #1a80c8; animation-delay: 0.19s; }

@keyframes cardPop {
    from { opacity: 0; transform: scale(0.88); }
    to   { opacity: 1; transform: scale(1); }
}

.result-card-top {
    padding: 16px 22px 12px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.result-card:nth-child(1) .result-card-top { background: #f0fbe8; }
.result-card:nth-child(2) .result-card-top { background: #fff5e8; }
.result-card:nth-child(3) .result-card-top { background: #e8f4ff; }

.result-card-icon { font-size: 28px; }
.result-card-label {
    font-family: 'Baloo 2', cursive;
    font-size: 14px;
    font-weight: 700;
    color: #5a3800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.result-card-body { padding: 16px 22px 20px; }

.result-value {
    font-family: 'Baloo 2', cursive;
    font-size: 36px;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 6px;
}

.result-card:nth-child(1) .result-value { color: #2e7d1a; }
.result-card:nth-child(2) .result-value { color: #a05a00; }
.result-card:nth-child(3) .result-value { color: #0a5a9e; }

.result-desc { font-size: 14px; color: #907050; line-height: 1.5; }

/* ════════════════════════════════
   FOOTER
════════════════════════════════ */
.footer {
    background: #2d1e0f;
    padding: 14px 52px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
}

.footer-brand {
    font-family: 'Baloo 2', cursive;
    font-size: 17px;
    font-weight: 700;
    color: #f5c040;
}

.footer-text {
    font-family: 'Baloo 2', cursive;
    font-size: 13px;
    color: rgba(255,230,180,0.45);
}

/* ════════════════════════════════
   RESPONSIVE
════════════════════════════════ */
@media (max-width: 960px) {
    .result-cards { grid-template-columns: 1fr; }
    .top-banner   { flex-direction: column; padding: 24px 22px 20px; min-height: unset; }
    .banner-icon  { display: none; }
    .step-guide   { padding: 10px 18px; }
    .form-wrapper { padding: 16px 18px; }
    .results-wrapper { padding: 14px 18px 24px; }
    .warn-box     { margin: 10px 18px 0; }
    .footer       { padding: 12px 18px; flex-direction: column; text-align: center; }
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

# ── Banner + Steps (pure HTML, no gap) ───────────────────────────────────────
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
<div class="form-wrapper">
""", unsafe_allow_html=True)

# ── Columns (inside the form-wrapper div opened above) ───────────────────────
st.markdown('<div class="cols-wrapper">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3, gap="medium")

# ── Card 1: Soil Nutrients ────────────────────────────────────────────────────
with col1:
    st.markdown("""
    <div class="form-card">
      <div class="card-top">
        <div class="card-emoji">🧪</div>
        <div>
          <div class="card-title">Soil Nutrients</div>
          <div class="card-hint">From your soil test report (kg/ha)</div>
        </div>
      </div>
      <div class="card-body">
    """, unsafe_allow_html=True)

    st.markdown('<span class="field-label first">Nitrogen (N)</span><span class="field-hint">Typical range: 0 – 140 kg/ha</span>', unsafe_allow_html=True)
    N = st.number_input("Nitrogen", min_value=0.0, max_value=500.0, value=None,
                        placeholder="e.g.  50", label_visibility="collapsed")

    st.markdown('<span class="field-label">Phosphorus (P)</span><span class="field-hint">Typical range: 0 – 145 kg/ha</span>', unsafe_allow_html=True)
    P = st.number_input("Phosphorus", min_value=0.0, max_value=500.0, value=None,
                        placeholder="e.g.  40", label_visibility="collapsed")

    st.markdown('<span class="field-label">Potassium (K)</span><span class="field-hint">Typical range: 0 – 205 kg/ha</span>', unsafe_allow_html=True)
    K = st.number_input("Potassium", min_value=0.0, max_value=500.0, value=None,
                        placeholder="e.g.  35", label_visibility="collapsed")

    st.markdown('</div></div>', unsafe_allow_html=True)

# ── Card 2: Weather & Water ───────────────────────────────────────────────────
with col2:
    st.markdown("""
    <div class="form-card">
      <div class="card-top">
        <div class="card-emoji">🌤️</div>
        <div>
          <div class="card-title">Weather &amp; Water</div>
          <div class="card-hint">Local climate conditions</div>
        </div>
      </div>
      <div class="card-body">
    """, unsafe_allow_html=True)

    st.markdown('<span class="field-label first">Temperature (°C)</span><span class="field-hint">Average daytime temperature</span>', unsafe_allow_html=True)
    temp = st.number_input("Temperature", min_value=-20.0, max_value=60.0, value=None,
                           placeholder="e.g.  28", label_visibility="collapsed")

    st.markdown('<span class="field-label">Humidity (%)</span><span class="field-hint">Relative humidity in the air</span>', unsafe_allow_html=True)
    humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=None,
                               placeholder="e.g.  65", label_visibility="collapsed")

    st.markdown('<span class="field-label">Rainfall (mm)</span><span class="field-hint">Annual or seasonal rainfall</span>', unsafe_allow_html=True)
    rainfall = st.number_input("Rainfall", min_value=0.0, value=None,
                               placeholder="e.g.  200", label_visibility="collapsed")

    st.markdown('<span class="field-label">Soil Moisture (%)</span><span class="field-hint">Current moisture in topsoil</span>', unsafe_allow_html=True)
    moisture = st.number_input("Moisture", min_value=0.0, max_value=100.0, value=None,
                               placeholder="e.g.  50", label_visibility="collapsed")

    st.markdown('</div></div>', unsafe_allow_html=True)

# ── Card 3: Soil Properties ───────────────────────────────────────────────────
with col3:
    st.markdown("""
    <div class="form-card">
      <div class="card-top">
        <div class="card-emoji">🌍</div>
        <div>
          <div class="card-title">Soil Properties</div>
          <div class="card-hint">Chemical quality of your soil</div>
        </div>
      </div>
      <div class="card-body">
    """, unsafe_allow_html=True)

    st.markdown('<span class="field-label first">Soil pH</span><span class="field-hint">0 = very acidic · 7 = neutral · 14 = alkaline</span>', unsafe_allow_html=True)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=None,
                         placeholder="e.g.  6.5", label_visibility="collapsed")

    st.markdown('<span class="field-label">Organic Carbon (%)</span><span class="field-hint">Organic matter content in soil</span>', unsafe_allow_html=True)
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=None,
                                     placeholder="e.g.  0.8", label_visibility="collapsed")

    st.markdown('<span class="field-label">Electrical Conductivity (dS/m)</span><span class="field-hint">Soil salinity · typical range: 0 – 4</span>', unsafe_allow_html=True)
    ec = st.number_input("Electrical Conductivity", min_value=0.0, value=None,
                         placeholder="e.g.  1.2", label_visibility="collapsed")

    predict_btn = st.button("🌱  Get Recommendation", use_container_width=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

# close cols-wrapper and form-wrapper
st.markdown('</div></div>', unsafe_allow_html=True)

# ── Validation & Prediction ───────────────────────────────────────────────────
if predict_btn:
    fields = {
        "Nitrogen (N)": N,
        "Phosphorus (P)": P,
        "Potassium (K)": K,
        "Temperature": temp,
        "Humidity": humidity,
        "Rainfall": rainfall,
        "Soil Moisture": moisture,
        "Soil pH": ph,
        "Organic Carbon": organic_carbon,
        "Electrical Conductivity": ec,
    }
    missing = [k for k, v in fields.items() if v is None]

    if missing:
        st.markdown(f"""
        <div class="warn-box">
            <span style="font-size:26px">⚠️</span>
            <span>Please fill in: <strong>{",  ".join(missing)}</strong></span>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing your soil data..."):
            crop = crop_model.predict([[N, P, K, temp, humidity, ph, rainfall]])[0]
            soil = soil_model.predict([[temp, humidity, moisture, ph, organic_carbon, ec, N, P, K]])[0]
            fert  = fert_model.predict([[temp, humidity, moisture, N, P, K]])[0]

        st.markdown(f"""
        <div class="results-wrapper">
            <div class="results-banner">
                <div class="results-banner-icon">✅</div>
                <div>
                    <div class="results-banner-text">Analysis Complete — Your Recommendations</div>
                    <div class="results-banner-sub">Based on your soil and weather inputs</div>
                </div>
            </div>
            <div class="result-cards">
                <div class="result-card">
                    <div class="result-card-top">
                        <div class="result-card-icon">🌾</div>
                        <div class="result-card-label">Best Crop to Grow</div>
                    </div>
                    <div class="result-card-body">
                        <div class="result-value">{crop}</div>
                        <div class="result-desc">Best suited for your soil nutrients and local climate.</div>
                    </div>
                </div>
                <div class="result-card">
                    <div class="result-card-top">
                        <div class="result-card-icon">🏔️</div>
                        <div class="result-card-label">Soil Type Detected</div>
                    </div>
                    <div class="result-card-body">
                        <div class="result-value">{soil}</div>
                        <div class="result-desc">Classified from moisture, pH, carbon &amp; salinity.</div>
                    </div>
                </div>
                <div class="result-card">
                    <div class="result-card-top">   
                        <div class="result-card-icon">🧴</div>
                        <div class="result-card-label">Fertilizer to Apply</div>
                    </div>
                    <div class="result-card-body">
                        <div class="result-value">{fert}</div>
                        <div class="result-desc">Corrects nutrient gaps and boosts field yield.</div>
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