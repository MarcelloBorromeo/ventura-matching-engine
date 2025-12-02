import streamlit as st
import pandas as pd
import time
from engine import InvestorDataPipeline, InvestorMatchingGraph

# -----------------------------------------------------------
# THEME + FONTS
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
.section-header {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #0E3A75 !important;
}
.card {
    background: rgba(255,255,255,0.75);
    padding: 22px;
    border-radius: 12px;
    border: 1.5px solid #AFCBFF;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# PAGE TITLE
# -----------------------------------------------------------
st.markdown("<h1 class='main-title'>Venture Investor Matching Engine</h1>", unsafe_allow_html=True)
st.markdown("<div class='subheader' style='text-align:center;color:#167DFF;'>IEC-powered precision investor recommendations</div>", unsafe_allow_html=True)


# -----------------------------------------------------------
# LOAD ENGINE
# -----------------------------------------------------------
@st.cache_resource
def load_engine():
    dp = InvestorDataPipeline("VC Backed IPO Data [Complete].csv")
    dp.generate_embeddings()
    key = st.secrets["OPENROUTER_API_KEY"]
    return InvestorMatchingGraph(dp, key)

engine = load_engine()


# -----------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------
with st.sidebar:
    st.header("Startup Profile")

    industry = st.text_input("Industry", "Software")

    deal = st.number_input("Deal Size ($M)", value=50.0, min_value=0.0, max_value=1_000_000_000.0)
    growth = st.number_input("Growth YoY", value=0.35, min_value=0.0, max_value=1.0)

    desc = st.text_area("Description", "AI workflow automation platform.")

    # ----------------------------
    # AI TOGGLE
    # ----------------------------
    st.subheader("AI Explanation")

    ai_explanation = st.checkbox("Enable AI-generated reasoning", value=True)

    toggle_color = "#167DFF" if ai_explanation else "#555555"
    toggle_label = "AI Explanation: ON" if ai_explanation else "AI Explanation: OFF"

    st.markdown(
        f"""
        <div style="
            background:{toggle_color};
            padding:10px;
            text-align:center;
            color:white;
            font-weight:700;
            border-radius:6px;">
            {toggle_label}
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------
# RUN MATCHING
# -----------------------------------------------------------
if st.button("Run Matching"):

    step_display = st.empty()
    def update_step(msg):
        step_display.markdown(f"<div class='progress-text'>{msg}</div>", unsafe_allow_html=True)

    with st.spinner("Running investor matching pipeline…"):
        results = engine.run(
            {
                "industry": industry,
                "deal_size_m": deal,
                "revenue_growth_yoy": growth,
                "description": desc,
            },
            progress_callback=update_step,
            use_llm=ai_explanation,
        )

    # Done message
    st.success("✔ Done")

    # -----------------------------------------------------------
    # TOP 3 MATCHES TABLE
    # -----------------------------------------------------------
    st.markdown("<div class='section-header'>Top 3 Matches</div>", unsafe_allow_html=True)

    df = pd.DataFrame(results)

    table_html = """
<style>
.match-table {
    width: 100%;
    border-collapse: collapse;
    background: rgba(255,255,255,0.80);
    border-radius: 12px;
    border: 1.5px solid #AFCBFF;
}
.match-table th {
    background:#167DFF;
    color:white;
    padding:12px;
    font-weight:700;
}
.match-table td {
    padding:12px;
    font-weight:600;
    text-align:center;
    color:#0E3A75;
}
</style>
<table class="match-table">
<tr><th>Investor</th><th>Match Score</th></tr>
"""
    for _, row in df.iterrows():
        table_html += f"<tr><td>{row['investor']}</td><td>{row['final']}</td></tr>"

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)

    # -----------------------------------------------------------
    # REASONING SECTION (ONLY IF AI IS ON)
    # -----------------------------------------------------------
    if ai_explanation:
        st.markdown("<div class='section-header'>Reasoning</div>", unsafe_allow_html=True)

        for r in results:
            st.markdown(f"""
            <div class='card'>
                <div class='investor-name'>{r['investor']}</div>
                <div class='explanation'>{r['explanation']}</div>
                <div class='web-summary'><strong>Web Summary:</strong> {r['web']}</div>
            </div>
            """, unsafe_allow_html=True)
