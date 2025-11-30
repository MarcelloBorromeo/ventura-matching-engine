import streamlit as st
import pandas as pd
from engine import InvestorDataPipeline, InvestorMatchingGraph

st.set_page_config(page_title="Investor Matcher", layout="wide", page_icon="🚀")

st.title("🚀 Venture Investor Matching Engine")

# -------------------------------------
# Load engine once
# -------------------------------------
@st.cache_resource
def load_engine():
    dp = InvestorDataPipeline("VC Backed IPO Data [Complete].csv")
    dp.generate_embeddings()
    key = st.secrets["OPENROUTER_API_KEY"]
    return InvestorMatchingGraph(dp, key)

engine = load_engine()

# -------------------------------------
# Sidebar
# -------------------------------------
with st.sidebar:
    st.header("Startup Profile")
    industry = st.text_input("Industry", "Software")
    deal = st.number_input("Deal Size ($M)", 50.0)
    growth = st.number_input("Growth YoY", 0.35)
    desc = st.text_area("Description", "AI-powered workflow automation.")

# -------------------------------------
# Spinner-only progress
# -------------------------------------
if st.button("🔍 Run Matching"):
    status = st.empty()  # spinner text

    def update(step):
        status.markdown(f"⬆️ **{step}**")

    with st.spinner("Running investor matching pipeline..."):
        results = engine.run(
            {
                "industry": industry,
                "deal_size_m": deal,
                "revenue_growth_yoy": growth,
                "description": desc
            },
            progress_callback=update
        )

    # -------------------------------------
    # Display results
    # -------------------------------------
    df = pd.DataFrame(results)
    st.subheader("🏆 Top Matches")
    st.dataframe(df[["investor", "final", "embedding"]], use_container_width=True)

    st.subheader("🧠 Explanations")
    for r in results:
        st.markdown(f"### {r['investor']} — Score {r['final']}")
        st.write(r["explanation"])
        st.write(f"**Web summary:**\n{r['web']}")
        st.markdown("---")
