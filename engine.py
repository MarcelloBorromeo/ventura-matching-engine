import json
import numpy as np
import pandas as pd
from typing import Dict, List, TypedDict
from sklearn.feature_extraction.text import TfidfVectorizer

import streamlit as st
from anthropic import Anthropic

# Load API key from Streamlit secrets
client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# ===================================================================
# STEP 1: DATA PIPELINE & EMBEDDINGS
# ===================================================================

class InvestorDataPipeline:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.investor_embeddings = {}
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.investor_fingerprint_vectors = {}
        self._clean_data()

    def _clean_data(self):
        self.df["Investor_Name"] = (
            self.df["Investor_Name"]
            .astype(str)
            .str.replace(r"[\r\n]+", " ", regex=True)
            .str.strip()
        )

        self.df["Business_Overview"] = self.df["Business_Overview"].fillna("")
        self.df["Competitive Analysis "] = self.df["Competitive Analysis "].fillna("")

    def create_investor_context(self, investor_name: str) -> str:
        investor_deals = self.df[self.df["Investor_Name"] == investor_name]
        companies = investor_deals["Portfolio_Company"].unique()
        industries = investor_deals["Industry"].unique()

        context = f"Investor: {investor_name}\n"
        context += f"Portfolio Companies: {', '.join(companies[:10])}\n"
        context += f"Industries: {', '.join(industries)}\n"
        context += f"Number of deals: {len(investor_deals)}\n\n"
        context += "Sample Company Profiles:\n"

        for _, row in investor_deals.head(3).iterrows():
            context += f"- {row['Portfolio_Company']}: {row['Business_Overview'][:200]}...\n"

        return context

    def create_investor_fingerprint(self, investor_name: str) -> str:
        investor_deals = self.df[self.df["Investor_Name"] == investor_name]
        parts = []

        companies = investor_deals["Portfolio_Company"].unique()
        industries = investor_deals["Industry"].unique()
        parts.append(f"Portfolio: {', '.join(companies)}")
        parts.append(f"Industries: {', '.join(industries)}")

        for t in investor_deals["Business_Overview"].dropna().unique():
            parts.append(t)
        for t in investor_deals["Competitive Analysis "].dropna().unique():
            parts.append(t)

        deal_sizes = investor_deals["Deal_Size_M"].dropna()
        if len(deal_sizes) > 0:
            parts.append(f"Typical deal size: ${deal_sizes.mean():.1f}M")

        return " ".join(" ".join(parts).split())

    def generate_embeddings(self):
        unique_investors = self.df["Investor_Name"].unique()

        investor_fingerprints = {}
        for investor in unique_investors:
            context = self.create_investor_context(investor)
            fingerprint = self.create_investor_fingerprint(investor)
            investor_fingerprints[investor] = fingerprint

            self.investor_embeddings[investor] = {
                "context": context,
                "fingerprint": fingerprint,
            }

        self.vectorizer.fit(investor_fingerprints.values())

        for inv in unique_investors:
            vec = self.vectorizer.transform([investor_fingerprints[inv]]).toarray()[0]
            self.investor_fingerprint_vectors[inv] = vec

    def calculate_embedding_similarity(self, startup_vector: np.ndarray, investor_name: str) -> float:
        investor_vector = self.investor_fingerprint_vectors[investor_name]

        dot = np.dot(startup_vector, investor_vector)
        norm_s = np.linalg.norm(startup_vector)
        norm_i = np.linalg.norm(investor_vector)
        if norm_s == 0 or norm_i == 0:
            return 0.0

        cosine_sim = dot / (norm_s * norm_i)
        return round(cosine_sim * 100, 2)


# ===================================================================
# STEP 2: LANGGRAPH-STYLE LLM REASONING
# ===================================================================

class GraphState(TypedDict):
    startup_profile: Dict
    all_investors: List[str]
    candidate_investors: List[Dict]
    ranked_results: List[Dict]


class InvestorMatchingGraph:
    def __init__(self, data_pipeline: InvestorDataPipeline):
        self.data = data_pipeline
        self.client = client

    # ---------------------- WEB CONTEXT NODE -----------------------
    def fetch_investor_web_context(self, investor_name: str) -> str:
        prompt = f"""
Perform a brief lookup for this investor and firm:

Investor: {investor_name}

Return 3–6 sentences about:
- Focus / thesis
- Stage and check size
- Geography
- Notable investments
- Any relevant public info
If unsure, state that.
"""

        try:
            resp = self.client.messages.create(
                model="claude-3.5-sonnet",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            return f"(Web context unavailable: {e})"

    # ---------------------- RETRIEVAL NODE -------------------------
    def retrieve_candidates(self, state: GraphState) -> GraphState:
        startup_text = f"{state['startup_profile']['industry']} {state['startup_profile']['description']}"
        startup_vector = self.data.vectorizer.transform([startup_text]).toarray()[0]

        scores = []
        for investor in state["all_investors"]:
            s = self.data.calculate_embedding_similarity(startup_vector, investor)
            scores.append({"investor_name": investor, "embedding_likelihood": s})

        scores.sort(key=lambda x: x["embedding_likelihood"], reverse=True)
        state["candidate_investors"] = scores[:3]
        return state

    # ---------------------- REASONING NODE -------------------------
    def reason_about_fit(self, state: GraphState) -> GraphState:
        startup = state["startup_profile"]

        for cand in state["candidate_investors"]:
            name = cand["investor_name"]
            inv_context = self.data.investor_embeddings[name]["context"]
            web_context = self.fetch_investor_web_context(name)

            prompt = f"""
Evaluate investor–startup fit.

STARTUP:
Industry: {startup['industry']}
Deal Size: {startup['deal_size_m']}
YOY Growth: {startup['revenue_growth_yoy']}
Description: {startup['description']}

INVESTOR (Dataset):
{inv_context}

INVESTOR (Web):
{web_context}

Embedding Score: {cand['embedding_likelihood']}

Return JSON:
{{
  "adjustment": -20 to 20,
  "explanation": "<2–3 sentence explanation>"
}}
"""

            try:
                resp = self.client.messages.create(
                    model="claude-3.5-sonnet",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                js = json.loads(resp.content[0].text)
                cand["llm_adjustment"] = js.get("adjustment", 0)
                cand["explanation"] = js.get("explanation", "")
                cand["web_context"] = web_context

            except Exception as e:
                cand["llm_adjustment"] = 0
                cand["explanation"] = f"LLM error: {e}"
                cand["web_context"] = web_context

        return state

    # ---------------------- RANKING NODE ---------------------------
    def rank_investors(self, state: GraphState) -> GraphState:
        for cand in state["candidate_investors"]:
            final_score = cand["embedding_likelihood"] + cand["llm_adjustment"]
            cand["final_score"] = max(0, min(100, round(final_score, 2)))

        state["ranked_results"] = sorted(
            state["candidate_investors"], key=lambda x: x["final_score"], reverse=True
        )
        return state

    # ---------------------- PIPELINE -------------------------------
    def run_pipeline(self, startup_profile: Dict):
        state = GraphState(
            startup_profile=startup_profile,
            all_investors=list(self.data.investor_embeddings.keys()),
            candidate_investors=[],
            ranked_results=[],
        )

        state = self.retrieve_candidates(state)
        state = self.reason_about_fit(state)
        state = self.rank_investors(state)

        return state["ranked_results"]
