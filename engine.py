import json
import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Callable, Optional, TypedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Helper
# -------------------------------
ProgressCallback = Optional[Callable[[str], None]]

def or_chat(model: str, messages: List[Dict], api_key: str, max_tokens=300):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise Exception(f"OpenRouter error {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"]


# -------------------------------
# TF-IDF Pipeline
# -------------------------------
class InvestorDataPipeline:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.vectorizer = TfidfVectorizer(max_features=600, ngram_range=(1,2))
        self.investor_vectors = {}
        self.investor_fingerprints = {}
        self._clean()

    def _clean(self):
        self.df["Investor_Name"] = (
            self.df["Investor_Name"]
            .astype(str).str.replace(r"[\r\n]+"," ",regex=True).str.strip()
        )
        self.df["Business_Overview"].fillna("", inplace=True)
        if "Competitive Analysis " in self.df.columns:
            self.df["Competitive Analysis "].fillna("", inplace=True)

    def _fingerprint(self, inv):
        rows = self.df[self.df["Investor_Name"] == inv]
        parts = []
        parts.append(" ".join(rows["Portfolio_Company"].dropna().astype(str).unique()))
        parts.append(" ".join(rows["Industry"].dropna().astype(str).unique()))
        parts.append(" ".join(rows["Business_Overview"].dropna().astype(str).unique()))
        if "Competitive Analysis " in self.df.columns:
            parts.append(" ".join(rows["Competitive Analysis "].dropna().astype(str).unique()))
        return " ".join(parts)

    def generate_embeddings(self):
        investors = self.df["Investor_Name"].unique()
        fps = {inv: self._fingerprint(inv) for inv in investors}
        self.investor_fingerprints = fps

        self.vectorizer.fit(fps.values())
        for inv in investors:
            self.investor_vectors[inv] = self.vectorizer.transform([fps[inv]]).toarray()[0]

    def similarity(self, startup_vec, inv):
        vec = self.investor_vectors[inv]
        sim = cosine_similarity(startup_vec.reshape(1,-1), vec.reshape(1,-1))[0][0]
        return round(sim*100, 2)


# -------------------------------
# Matching Engine (Retrieval + LLM Reasoning)
# -------------------------------
class SuperInvestorMatcher:
    def __init__(self, pipeline: InvestorDataPipeline, key: str):
        self.data = pipeline
        self.key = key
        self.model = "anthropic/claude-3.5-sonnet"

    # ------------------- Retrieval -------------------
    def retrieve_candidates(self, startup_profile: dict) -> List[Dict]:
        txt = f"{startup_profile['industry']} {startup_profile['description']}"
        vec = self.data.vectorizer.transform([txt]).toarray()[0]

        scores = []
        for inv in self.data.investor_vectors.keys():
            sim = self.data.similarity(vec, inv)
            scores.append({"investor": inv, "score": sim})

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:3]

    # ------------------- Web Context -------------------
    def fetch_web(self, investor: str) -> str:
        sys = "Return 2 short bullet points describing this investor's focus + stage. <=10 words each."
        usr = f"Investor: {investor}"
        try:
            return or_chat(self.model,
                [{"role":"system","content":sys},{"role":"user","content":usr}],
                self.key, max_tokens=120
            ).strip()
        except:
            return "(web unavailable)"

    # ------------------- LLM Reasoning -------------------
    def reason(self, startup: dict, cand: dict, context: str, web: str) -> dict:
        sys = (
            "You are a concise VC analyst. Write one paragraph (max 4 sentences) "
            "explaining the startup–investor fit using embedding score, dataset context, and web summary."
        )

        usr = (
            f"STARTUP:\nIndustry:{startup['industry']}\nDeal:{startup['deal_size_m']}\n"
            f"Growth:{startup['revenue_growth_yoy']}\nDesc:{startup['description']}\n\n"
            f"SIMILARITY SCORE: {cand['score']}\n\n"
            f"INVESTOR DATASET CONTEXT:\n{context}\n\nWEB SUMMARY:\n{web}"
        )

        try:
            explanation = or_chat(
                self.model,
                [{"role":"system","content":sys},{"role":"user","content":usr}],
                self.key,
                max_tokens=250
            )
        except Exception as e:
            explanation = f"(LLM error: {e})"

        cand["explanation"] = explanation.strip()
        cand["web"] = web
        return cand

    # ------------------- Full Pipeline -------------------
    def run_pipeline(self, startup: dict, progress_cb: ProgressCallback=None) -> List[Dict]:

        if progress_cb: progress_cb("Retrieving investors…")
        cands = self.retrieve_candidates(startup)

        results = []
        for c in cands:
            inv = c["investor"]
            if progress_cb: progress_cb(f"Finding web summary for {inv}…")
            web = self.fetch_web(inv)

            ctx = self.data.investor_fingerprints[inv]
            if progress_cb: progress_cb(f"Analyzing fit for {inv}…")
            out = self.reason(startup, c, ctx, web)
            results.append(out)

        if progress_cb: progress_cb("Done.")
        return results
