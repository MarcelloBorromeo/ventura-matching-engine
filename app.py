import streamlit as st
import pandas as pd
import time

from engine import InvestorDataPipeline, InvestorMatchingGraph

# -----------------------------------------------------------
# IEC PREMIUM UI THEME — UPDATED FONTS (Montserrat + Open Sans)
# -----------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700;800;900&family=Open+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>
html, body, div, span, textarea, input, button {
    font-family: 'Open Sans', sans-serif !important;
}

.main-title, .section-header, .investor-name, .stButton>button {
    font-family: 'Montserrat', sans-serif !important;
}

body, .stApp {
    background: linear-gradient(180deg, #E8F1FF 0%, #FFFFFF 60%) !important;
}

.block-container {
    max-width: 1000px !important;
}

.main-title {
    text-align: center;
    font-size: 46px !important;
    font-weight: 800 !important;
    color: #000000 !important;
    margin-bottom: 5px;
}

.subheader {
    text-align: center;
    font-size: 20px;
    color: #167DFF;
    margin-bottom: 20px;
}

.section-header {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #0E3A75 !important;
    margin-top: 10px;
}

.card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(8px);
    border-radius: 12px;
    padding: 22px;
    margin-top: 10px;
    border: 1.5px solid #AFCBFF;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
}

.investor-name {
    font-size: 22px;
    font-weight: 700;
    color: #167DFF;
}

.explanation {
    text-align: justify;
    color: #0E3A75;
    font-size: 16px;
    margin-bottom: 12px;
}

.web-summary {
    font-size: 14px;
    color: #167DFF;
    font-weight: 600;
}

.stButton>button {
    background: #167DFF;
    color: white;
    border-radius: 8px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: 700 !important;
}
.stButton>button:hover {
    background: #0E3A75;
}

.progress-line {
    display: flex;
    align-items: center;
    color: #0E3A75;
    font-size: 15px;
    margin-top: 8px;
}

.loader {
    border: 3px solid #d0e3ff;
    border-top: 3px solid #167DFF;
    border-radius: 50%;
    width: 14px;
    height: 14px;
    animation: spin 1s linear infinite;
    margin-right: 10px;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# Title
# -----------------------------------------------------------
st.markdown("<h1 class='main-title'>Venture Investor Matching Engine</h1>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>IEC-powered precision investor recommendations</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# Load Engine
# -----------------------------------------------------------
@st.cache_resource
def load_engine():
    dp = InvestorDataPipeline("VC Backed IPO Data [Complete].csv")
    dp.generate_embeddings()
    key = st.secrets["OPENROUTER_API_KEY"]
    return InvestorMatchingGraph(dp, key)

engine = load_engine()

# -----------------------------------------------------------
# Sidebar
# -----------------------------------------------------------
with st.sidebar:
    st.header("Startup Profile")

    industry = st.text_input("Industry", "Software")
    deal = st.number_input("Deal Size ($M)", value=50.0)
    growth = st.number_input("Growth YoY", value=0.35)
    desc = st.text_area("Description", "AI workflow automation platform.")

# -----------------------------------------------------------
# Run Matching
# -----------------------------------------------------------
if st.button("Run Matching"):

    step_display = st.empty()

    def update_step(text):
        step_display.markdown(
            f"""
<div class="progress-line">
    <div class="loader"></div>
    <span>{text}</span>
</div>
            """,
            unsafe_allow_html=True,
        )

    with st.spinner("Running investor matching pipeline…"):
        results = engine.run(
            {
                "industry": industry,
                "deal_size_m": deal,
                "revenue_growth_yoy": growth,
                "description": desc,
            },
            progress_callback=update_step,
        )

        done_msg = st.empty()
        done_msg.markdown(
            "<div class='progress-line' style='color:green;'>✔ Done</div>",
            unsafe_allow_html=True,
        )
        time.sleep(2)
        done_msg.empty()

    # -----------------------------------------------------------
    # Top 3 MATCHES TABLE (PATCHED — HTML LEFT-ALIGNED)
    # -----------------------------------------------------------
    st.markdown("<div class='section-header'>Top 3 Matches</div>", unsafe_allow_html=True)

    df = pd.DataFrame(results)

    # *** IMPORTANT — MUST BE LEFT-ALIGNED ***
    table_html = """
<style>
.match-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
    background: rgba(255, 255, 255, 0.80);
    border-radius: 12px;
    overflow: hidden;
    border: 1.5px solid #AFCBFF;
    font-family: 'Open Sans', sans-serif !important;
}
.match-table th {
    background-color: #167DFF;
    color: white;
    font-size: 18px;
    padding: 14px;
}
.match-table td {
    padding: 14px;
    text-align: center;
    font-size: 17px;
    color: #0E3A75;
}
</style>

<table class="match-table">
<tr>
    <th>Investor</th>
    <th>Match Score</th>
</tr>
"""

    for _, row in df.iterrows():
        table_html += f"""
<tr>
    <td><strong>{row['investor']}</strong></td>
    <td><strong>{row['final']}</strong></td>
</tr>
"""

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)

    # -----------------------------------------------------------
    # Reasoning
    # -----------------------------------------------------------
    st.markdown("<div class='section-header'>Reasoning</div>", unsafe_allow_html=True)

    for r in results:
        st.markdown(
            f"""
<div class='card'>
    <div class='investor-name'>{r['investor']}</div>
    <div class='explanation'>{r['explanation']}</div>
    <div class='web-summary'><strong>Web Summary:</strong> {r['web']}</div>
</div>
""",
            unsafe_allow_html=True,
        )
