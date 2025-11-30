import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
from typing import Dict, List, TypedDict
from sklearn.feature_extraction.text import TfidfVectorizer


# ================================================================
# OPENROUTER Chat Completion Helper
# ================================================================
def or_chat(model: str, messages: List[Dict], api_key: str, max_tokens: int = 300):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)

    if resp.status_code != 200:
        raise Exception(f"OpenRouter Error {resp.status_code}: {resp.text}")

    return resp.json()["choices"][0]["message"]["content"]


# ================================================================
# DATA PIPELINE (TF-IDF Embeddings)
# ================================================================
class InvestorDataPipeline:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.vectorizer = TfidfVectorizer(max_features=600, ngram_range=(1, 2))
        self.investor_embeddings = {}
        self.investor_vectors = {}
        self._clean_data()

    def _clean_data(self):
        self.df["Investor_Name"] = (
            self.df["Investor_Name"].astype(str).str.replace(r"[\r\n]+", " ", regex=True).str.strip()
        )
        self.df["Business_Overview"].fillna("", inplace=True)
        if "Competitive Analysis " in self.df.columns:
            self.df["Competitive Analysis "].fillna("", inplace=True)

    def _fingerprint(self, inv: str) -> str:
        deals = self.df[self.df["Investor_Name"] == inv]
        text = " ".join([
            " ".join(deals["Portfolio_Company"].astype(str).unique()),
            " ".join(deals["Industry"].astype(str).unique()),
            " ".join(deals["Business_Overview"].astype(str).unique()),
            " ".join(deals.get("Competitive Analysis ", "").astype(str).unique()),
        ])
        return text

    def generate_embeddings(self):
        investors = self.df["Investor_Name"].unique()
        fps = {inv: self._fingerprint(inv) for inv in investors}
        self.vectorizer.fit(fps.values())

        for inv in investors:
            vec = self.vectorizer.transform([fps[inv]]).toarray()[0]
            self.investor_vectors[inv] = vec
            self.investor_embeddings[inv] = fps[inv]

    def similarity(self, startup_vec, inv):
        v2 = self.investor_vectors[inv]
        dot = np.dot(startup_vec, v2)
        n1, n2 = np.linalg.norm(startup_vec), np.linalg.norm(v2)
        return 0 if n1 == 0 or n2 == 0 else round((dot / (n1 * n2)) * 100, 2)


# ================================================================
# LANGGRAPH-LIKE MATCHING ENGINE
# ================================================================
class GraphState(TypedDict):
    startup_profile: Dict
    investor_names: List[str]
    candidates: List[Dict]
    ranked: List[Dict]


class InvestorMatchingGraph:
    def __init__(self, pipeline: InvestorDataPipeline, key: str):
        self.data = pipeline
        self.key = key
        self.model = "anthropic/claude-3.5-sonnet"

    # ------------------------------------------------------------
    def retrieve(self, state: GraphState):
        txt = f"{state['startup_profile']['industry']} {state['startup_profile']['description']}"
        vec = self.data.vectorizer.transform([txt]).toarray()[0]

        scores = []
        for inv in state["investor_names"]:
            scores.append({
                "investor": inv,
                "embedding": self.data.similarity(vec, inv)
            })

        scores.sort(key=lambda x: x["embedding"], reverse=True)
        state["candidates"] = scores[:5]
        return state

    # ------------------------------------------------------------
    def fetch_web(self, name: str) -> str:
        sys = "Return 2 short bullet points summarizing this investor's focus and stage. Keep each under 10 words."
        usr = f"Investor name: {name}"
        try:
            out = or_chat(self.model,
                          [{"role": "system", "content": sys},
                           {"role": "user", "content": usr}],
                          self.key, max_tokens=120)
            return out.strip()
        except Exception as e:
            return f"(web unavailable: {e})"

    # ------------------------------------------------------------
    def reason(self, state: GraphState):
        startup = state["startup_profile"]

        for c in state["candidates"]:
            inv = c["investor"]
            web = self.fetch_web(inv)

            sys = (
                "You are a concise VC analyst. Write 1 paragraph (max 4 sentences) "
                "explaining the fit between this startup and investor. "
                "Be specific, factual, and rely on the dataset, web signals, and embedding score."
            )

            usr = (
                f"STARTUP:\nIndustry: {startup['industry']}\nDeal Size: {startup['deal_size_m']}\n"
                f"Growth: {startup['revenue_growth_yoy']}\nDescription: {startup['description']}\n\n"
                f"INVESTOR EMBEDDING SCORE: {c['embedding']}\n"
                f"WEB SUMMARY:\n{web}"
            )

            try:
                explanation = or_chat(self.model,
                                      [{"role": "system", "content": sys},
                                       {"role": "user", "content": usr}],
                                      self.key, max_tokens=180)
                c["explanation"] = explanation.strip()
                c["web"] = web
                c["final"] = max(0, min(100, c["embedding"]))
            except Exception as e:
                c["explanation"] = f"(LLM error: {e})"
                c["web"] = web
                c["final"] = c["embedding"]

        return state

    # ------------------------------------------------------------
    def rank(self, state: GraphState):
        state["ranked"] = sorted(state["candidates"], key=lambda x: x["final"], reverse=True)
        return state

    # ------------------------------------------------------------
    def run(self, startup: Dict, progress_callback):
        state = GraphState(
            startup_profile=startup,
            investor_names=list(self.data.investor_vectors.keys()),
            candidates=[],
            ranked=[]
        )

        progress_callback("Retrieving candidates…")
        state = self.retrieve(state)

        progress_callback("Fetching web summaries…")
        state = self.reason(state)

        progress_callback("Ranking results…")
        state = self.rank(state)

        progress_callback("Done.")
        return state["ranked"]
