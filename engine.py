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
# DATA PIPELINE
# ================================================================
class InvestorDataPipeline:

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.vectorizer = TfidfVectorizer(max_features=600, ngram_range=(1, 2))
        self.investor_vectors = {}
        self.investor_contexts = {}
        self._clean_data()

    def _clean_data(self):
        self.df["Investor_Name"] = (
            self.df["Investor_Name"]
            .astype(str)
            .str.replace(r"[\r\n]+", " ", regex=True)
            .str.strip()
        )

        self.df["Business_Overview"].fillna("", inplace=True)

        if "Competitive Analysis " in self.df.columns:
            self.df["Competitive Analysis "].fillna("", inplace=True)

    def _fingerprint(self, investor: str) -> str:
        deals = self.df[self.df["Investor_Name"] == investor]
        return " ".join([
            " ".join(deals["Portfolio_Company"].astype(str).unique()),
            " ".join(deals["Industry"].astype(str).unique()),
            " ".join(deals["Business_Overview"].astype(str).unique()),
            " ".join(deals.get("Competitive Analysis ", "").astype(str).unique()),
        ])

    def _context(self, investor: str) -> str:
        deals = self.df[self.df["Investor_Name"] == investor]

        ctx = f"Investor: {investor}\n"
        ctx += f"Industries: {', '.join(deals['Industry'].unique())}\n"
        ctx += f"Portfolio: {', '.join(deals['Portfolio_Company'].unique()[:10])}\n"
        ctx += f"Deals: {len(deals)}"

        return ctx

    def generate_embeddings(self):
        investors = self.df["Investor_Name"].unique()
        fps = {inv: self._fingerprint(inv) for inv in investors}

        self.vectorizer.fit(fps.values())

        for inv in investors:
            vec = self.vectorizer.transform([fps[inv]]).toarray()[0]
            self.investor_vectors[inv] = vec
            self.investor_contexts[inv] = self._context(inv)

    def similarity(self, startup_vec, investor):
        v2 = self.investor_vectors[investor]
        dot = np.dot(startup_vec, v2)
        n1, n2 = np.linalg.norm(startup_vec), np.linalg.norm(v2)

        return 0 if n1 == 0 or n2 == 0 else round((dot / (n1 * n2)) * 100, 2)


# ================================================================
# MATCHING ENGINE
# ================================================================
class GraphState(TypedDict):
    startup: Dict
    investors: List[str]
    candidates: List[Dict]
    ranked: List[Dict]


class InvestorMatchingGraph:

    def __init__(self, pipeline: InvestorDataPipeline, key: str):
        self.data = pipeline
        self.key = key
        self.model = "anthropic/claude-3.5-sonnet"

    # ------------------------------------------------------------
    def retrieve(self, state: GraphState):
        txt = f"{state['startup']['industry']} {state['startup']['description']}"
        vec = self.data.vectorizer.transform([txt]).toarray()[0]

        scores = []
        for inv in state["investors"]:
            scores.append({
                "investor": inv,
                "embedding": self.data.similarity(vec, inv)
            })

        scores.sort(key=lambda x: x["embedding"], reverse=True)
        state["candidates"] = scores[:3]  # Top 3 matches only
        return state

    # ------------------------------------------------------------
    def fetch_web(self, investor):
        sys_prompt = (
            "Return 2 short bullet points (<=10 words each) "
            "describing this investor's focus, stage, or check size."
        )

        try:
            return or_chat(
                self.model,
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Investor: {investor}"},
                ],
                self.key,
                max_tokens=120
            )
        except Exception as e:
            return f"(web unavailable: {e})"

    # ------------------------------------------------------------
    def reason(self, state: GraphState):
        s = state["startup"]

        for c in state["candidates"]:
            inv = c["investor"]
            web = self.fetch_web(inv)
            dataset_context = self.data.investor_contexts[inv]

            system_msg = (
                "You are a concise VC analyst and web researcher. Write one paragraph (max 4 sentences) "
                "explaining the investor–startup fit using evidence from the embedding score, "
                "dataset context, web summary, and YOUR own web search. Be specific, insightful, and avoid generic language."
            )

            user_msg = (
                f"STARTUP:\n"
                f"Industry: {s['industry']}\n"
                f"Deal Size: ${s['deal_size_m']}M\n"
                f"Growth: {s['revenue_growth_yoy']}\n"
                f"Description: {s['description']}\n\n"
                f"INVESTOR EMBEDDING SCORE: {c['embedding']}\n"
                f"WEB SUMMARY:\n{web}\n\n"
                f"DATASET CONTEXT:\n{dataset_context}"
            )

            try:
                explanation = or_chat(
                    self.model,
                    [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    self.key,
                    max_tokens=220
                )

                c["explanation"] = explanation.strip()
                c["web"] = web
                c["final"] = c["embedding"]

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
    def run(self, startup, progress_callback):
        state = GraphState(
            startup=startup,
            investors=list(self.data.investor_vectors.keys()),
            candidates=[],
            ranked=[]
        )

        progress_callback("Retrieving candidates…")
        state = self.retrieve(state)

        progress_callback("Analyzing fit…")
        state = self.reason(state)

        progress_callback("Ranking…")
        state = self.rank(state)

        progress_callback("✅ Done")
        return state["ranked"]
