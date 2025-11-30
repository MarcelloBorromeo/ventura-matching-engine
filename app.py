import streamlit as st
import pandas as pd
from engine import InvestorDataPipeline, InvestorMatchingGraph

# -----------------------------------------------------------
# Custom IEC Theme CSS
# -----------------------------------------------------------
st.markdown("""
<style>
    /* Centered Title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #167DFF;
        margin-top: -20px;
        letter-spacing: 1px;
    }

    /* Section Headers */
    .section-header {
        font-size: 26px;
        font-weight: 600;
        color: #0E3A75;
        margin-top: 25px;
    }

    /* Card-style Containers */
    .result-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E6F0FF;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #167DFF;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0E3A75;
        color: #ffffff;
    }

    /* Table Styling */
    table {
        border-collapse: collapse;
        width: 100%;
        background: white;
    }
    th {
        background-color: #167DFF !important;
        color: white !important;
        text-align: left !important;
        font-size: 16px !important;
        padding: 8px !important;
    }
    td {
        background-color: #F8FAFF;
        padding: 8px !important;
        font-size: 15px !important;
    }

    /* Text */
    .explanation-text {
        font-size: 16px;
        color: #0E3A75;
        line-height: 1.45;
        margin-top: 8px;
    }

    .web-summary {
        font-size: 14px;
        color: #167DFF;
        margin-top: -4px;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# Title
# -----------------------------------------------------------
st.markdown("<h1 class='main-title'>Venture Investor Matching Engine</h1>", unsafe_allow_html=True)

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
# Sidebar Inputs
# -----------------------------------------------------------
with st.sidebar:
    st.header("Startup Profile")
    industry = st.text_input("Industry", "Software")
    deal = st.number_input("Deal Size ($M)", 50.0)
    growth = st.number_input("Revenue Growth YoY", 0.35)
    desc = st.text_area("Description", "AI workflow automation platform.")


# -----------------------------------------------------------
# Matching Button
# -----------------------------------------------------------
if st.button("Run Matching"):
    status = st.empty()

    def update_step(text):
        status.markdown(f"**{text}**")

    with st.spinner("Running investor matching pipeline…"):
        results = engine.run(
            {
                "industry": industry,
                "deal_size_m": deal,
                "revenue_growth_yoy": growth,
                "description": desc
            },
            progress_callback=update_step
        )

    status.markdown("**<span style='color:green;'>✔ Done</span>**", unsafe_allow_html=True)

    df = pd.DataFrame(results)

    # -----------------------------------------------------------
    # Results (Top 3 Investors)
    # -----------------------------------------------------------
    st.markdown("<div class='section-header'>Top 3 Matches</div>", unsafe_allow_html=True)

    st.dataframe(
        df[["investor", "final"]].rename(columns={
            "investor": "Investor",
            "final": "Match Score"
        }),
        use_container_width=True
    )

    # -----------------------------------------------------------
    # Explanations
    # -----------------------------------------------------------
    st.markdown("<div class='section-header'>Reasoning</div>", unsafe_allow_html=True)

    for r in results:
        st.markdown(f"<div class='result-card'><h4 style='color:#167DFF;'>{r['investor']}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p class='explanation-text'>{r['explanation']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='web-summary'><strong>Web Summary:</strong> {r['web']}</p></div>", unsafe_allow_html=True)
