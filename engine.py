import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
from typing import Dict, List, TypedDict, Callable
from sklearn.feature_extraction.text import TfidfVectorizer

# --- OpenRouter chat helper (OpenAI-style) ---
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
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# --- Data pipeline (embeddings are TF-IDF fingerprint vectors) ---
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
        self.df["Business_Overview"].fillna("", inplace=True)
        # keep exact header name from CSV (if trailing space exists)
        if "Competitive Analysis " in self.df.columns:
            self.df["Competitive Analysis "].fillna("", inplace=True)
        else:
            # fallback if no trailing space column
            self.df["Competitive Analysis"] = self.df.get("Competitive Analysis", "")
            self.df["Competitive Analysis"].fillna("", inplace=True)

    def create_investor_context(self, investor_name: str) -> str:
        deals = self.df[self.df["Investor_Name"] == investor_name]
        companies = deals["Portfolio_Company"].unique()
        industries = deals["Industry"].unique()
        ctx = f"Investor: {investor_name}\nPortfolio Companies: {', '.join(companies[:10])}\nIndustries: {', '.join(industries)}\nDeals: {len(deals)}\n\n"
        ctx += "Sample Profiles:\n"
        for _, row in deals.head(3).iterrows():
            ctx += f"- {row['Portfolio_Company']}: {row['Business_Overview'][:200]}...\n"
        return ctx

    def create_investor_fingerprint(self, investor_name: str) -> str:
        deals = self.df[self.df["Investor_Name"] == investor_name]
        parts = []
        parts.append(f"Portfolio: {', '.join(deals['Portfolio_Company'].unique())}")
        parts.append(f"Industries: {', '.join(deals['Industry'].unique())}")
        for t in deals["Business_Overview"].dropna().unique():
            parts.append(t)
        comp_col = "Competitive Analysis "
        if comp_col in self.df.columns:
            for t in deals[comp_col].dropna().unique():
                parts.append(t)
        deal_sizes = deals.get("Deal_Size_M", pd.Series()).dropna()
        if len(deal_sizes) > 0:
            parts.append(f"Typical deal size: ${deal_sizes.mean():.1f}M")
        return " ".join(" ".join(parts).split())

    def generate_embeddings(self):
        investors = self.df["Investor_Name"].unique()
        fingerprints = {}
        for inv in investors:
            fp = self.create_investor_fingerprint(inv)
            ctx = self.create_investor_context(inv)
            fingerprints[inv] = fp
            self.investor_embeddings[inv] = {"fingerprint": fp, "context": ctx}
        # fit TF-IDF to investor fingerprints
        self.vectorizer.fit(fingerprints.values())
        for inv in investors:
            vec = self.vectorizer.transform([fingerprints[inv]]).toarray()[0]
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


# --- Graph pipeline and run with progress callback ---
class GraphState(TypedDict):
    startup_profile: Dict
    all_investors: List[str]
    candidate_investors: List[Dict]
    ranked_results: List[Dict]


# Progress callback signature:
# progress_callback(step_name: str, detail: str, percent: int) -> None
ProgressCallback = Callable[[str, str, int], None]


class InvestorMatchingGraph:
    """Pipeline that supports progress reporting via a callback."""

    def __init__(self, data_pipeline: InvestorDataPipeline, openrouter_key: str):
        self.data = data_pipeline
        self.openrouter_key = openrouter_key
        # set default model; can be changed externally if needed
        self.model = "anthropic/claude-3.5-sonnet"

    def _report(self, cb: ProgressCallback, step: str, detail: str, pct: int):
        try:
            if cb:
                cb(step, detail, pct)
        except Exception:
            # progress callback must never break pipeline
            pass

    # Node 1: retrieve candidates (embedding-based)
    def retrieve_candidates(self, state: GraphState, cb: ProgressCallback = None) -> GraphState:
        self._report(cb, "Retrieval", "Computing similarity scores...", 10)
        startup_text = f"{state['startup_profile']['industry']} {state['startup_profile']['description']}"
        startup_vector = self.data.vectorizer.transform([startup_text]).toarray()[0]
        scores = []
        for investor in state["all_investors"]:
            s = self.data.calculate_embedding_similarity(startup_vector, investor)
            scores.append({"investor_name": investor, "embedding_likelihood": s})
        scores.sort(key=lambda x: x["embedding_likelihood"], reverse=True)
        state["candidate_investors"] = scores[:5]  # top 5 candidates for more robust reasoning
        self._report(cb, "Retrieval", f"Top {len(state['candidate_investors'])} candidates selected", 20)
        return state

    # helper: concise web summary (3 bullet points)
    def fetch_investor_web_context(self, investor_name: str, cb: ProgressCallback = None) -> str:
        self._report(cb, "Web Enrichment", f"Fetching web summary for {investor_name}", 30)
        sys_msg = (
            "You are a concise web researcher. Return a tight summary in 3 short bullet points about the investor's focus, typical stage/check size, and notable investments or geography. "
            "Keep each bullet to <= 12 words. No commentary or explanation—just bullets."
        )
        user_prompt = f"Investor name: {investor_name}\nProvide 3 short bullets."
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_prompt},
        ]
        try:
            out = or_chat(self.model, messages, api_key=self.openrouter_key, max_tokens=200)
            self._report(cb, "Web Enrichment", f"Fetched web summary for {investor_name}", 40)
            return out.strip()
        except Exception as e:
            self._report(cb, "Web Enrichment", f"Web failure for {investor_name}: {e}", 40)
            return f"(Web context unavailable: {e})"

    # Node 2: LLM reasoning; returns adjustment and explanation
    def reason_about_fit(self, state: GraphState, cb: ProgressCallback = None) -> GraphState:
        self._report(cb, "Reasoning", "Starting LLM analysis for candidates...", 45)

        startup = state["startup_profile"]
        total = len(state["candidate_investors"])
        for idx, cand in enumerate(state["candidate_investors"], start=1):
            name = cand["investor_name"]
            inv_context = self.data.investor_embeddings.get(name, {}).get("context", "")
            web_context = self.fetch_investor_web_context(name, cb=cb)
            cand["web_context"] = web_context

            # strict system prompt for concise, numbered JSON output
            system_msg = (
                "You are a concise venture investment analyst. Analyze fit quickly and precisely. "
                "Produce ONLY a JSON object with exactly two keys: 'adjustment' (integer between -20 and 20) and 'explanation' (one sentence, <= 30 words). "
                "Do not include any extra text, punctuation, or diagnostics. Be evidence-based and specific."
            )

            user_msg = (
                f"STARTUP:\nIndustry: {startup['industry']}\nDeal Size: ${startup.get('deal_size_m', 'N/A')}M\n"
                f"Growth: {startup.get('revenue_growth_yoy', 'N/A')}\nDescription: {startup['description']}\n\n"
                f"INVESTOR_DATASET_CONTEXT:\n{inv_context}\n\nWEB_CONTEXT:\n{web_context}\n\n"
                f"OBJECTIVE_EMBEDDING_SCORE: {cand['embedding_likelihood']}"
            )

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            try:
                raw = or_chat(self.model, messages, api_key=self.openrouter_key, max_tokens=220)
                # try to locate JSON in response (be permissive)
                txt = raw.strip()
                js = None
                # If exact JSON, parse directly
                try:
                    js = json.loads(txt)
                except Exception:
                    # attempt to find braces
                    start = txt.find("{")
                    end = txt.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        try:
                            js = json.loads(txt[start:end+1])
                        except Exception:
                            js = None
                if js and "adjustment" in js:
                    adj = int(round(float(js.get("adjustment", 0))))
                    adj = max(-20, min(20, adj))
                    cand["llm_adjustment"] = adj
                    expl = js.get("explanation", "")
                    cand["explanation"] = expl.strip()
                else:
                    # fallback safe values
                    cand["llm_adjustment"] = 0
                    cand["explanation"] = f"LLM output could not be parsed. Raw: {txt[:200]}"
                self._report(cb, "Reasoning", f"Analyzed {idx}/{total}: {name}", 60 + int(30 * idx / max(total,1)))
            except Exception as e:
                cand["llm_adjustment"] = 0
                cand["explanation"] = f"LLM error: {e}"
                self._report(cb, "Reasoning", f"LLM error on {name}: {e}", 60 + int(30 * idx / max(total,1)))
        return state

    # Node 3: ranking
    def rank_investors(self, state: GraphState, cb: ProgressCallback = None) -> GraphState:
        self._report(cb, "Ranking", "Computing final scores and ordering...", 95)
        for cand in state["candidate_investors"]:
            final_score = cand["embedding_likelihood"] + cand.get("llm_adjustment", 0)
            cand["final_score"] = max(0, min(100, round(final_score, 2)))
        state["ranked_results"] = sorted(state["candidate_investors"], key=lambda x: x["final_score"], reverse=True)
        self._report(cb, "Ranking", "Ranking complete", 100)
        return state

    # run pipeline with optional progress callback
    def run_pipeline(self, startup_profile: Dict, cb: ProgressCallback = None) -> List[Dict]:
        state = GraphState(
            startup_profile=startup_profile,
            all_investors=list(self.data.investor_embeddings.keys()),
            candidate_investors=[],
            ranked_results=[]
        )
        state = self.retrieve_candidates(state, cb=cb)
        state = self.reason_about_fit(state, cb=cb)
        state = self.rank_investors(state, cb=cb)
        return state["ranked_results"]
