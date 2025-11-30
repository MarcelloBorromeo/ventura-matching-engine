# engine.py
import json
import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Callable, TypedDict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Type for progress callback: progress_callback(step_text: str) -> None
ProgressCallback = Optional[Callable[[str], None]]

# -------------------------
# OpenRouter helper
# -------------------------
def or_chat(model: str, messages: List[Dict], api_key: str, max_tokens: int = 300, timeout: int = 60) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise Exception(f"OpenRouter Error {resp.status_code}: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]


# -------------------------
# Internal typed state
# -------------------------
class GraphState(TypedDict):
    startup_profile: Dict
    all_investors: List[str]
    candidates: List[Dict]
    ranked_results: List[Dict]


# -------------------------
# Hybrid Data Pipeline
# -------------------------
class HybridDataPipeline:
    """
    Builds both:
     - TF-IDF text fingerprint vectors per investor
     - Numeric fingerprints per investor using StandardScaler + OneHotEncoder
    """
    def __init__(self, csv_path: str, text_cols: List[str] = None, numeric_cols: List[str] = None, cat_cols: List[str] = None):
        self.df = pd.read_csv(csv_path)
        # default text columns to build TF-IDF fingerprint
        if text_cols is None:
            text_cols = ["Portfolio_Company", "Industry", "Business_Overview", "Competitive Analysis "]
        self.text_cols = [c for c in text_cols if c in self.df.columns]
        # default numeric columns
        if numeric_cols is None:
            numeric_cols = ["Deal_Size_M", "Revenue_LTM_M", "Revenue_Growth_YoY", "Gross_Margin"]
        self.numeric_cols = [c for c in numeric_cols if c in self.df.columns]
        # categorical columns for numeric pipeline
        if cat_cols is None:
            cat_cols = ["Industry"]
        self.cat_cols = [c for c in cat_cols if c in self.df.columns]

        # placeholders
        self.vectorizer = TfidfVectorizer(max_features=800, ngram_range=(1,2))
        self.investor_text_vectors: Dict[str, np.ndarray] = {}
        self.investor_text_fingerprints: Dict[str, str] = {}

        # numeric preprocessor will be created in generate_numeric_profiles()
        self.preprocessor = None
        self.investor_numeric_profiles = None  # pandas.DataFrame indexed by investor

        self._clean_names()

    def _clean_names(self):
        if "Investor_Name" in self.df.columns:
            self.df["Investor_Name"] = (
                self.df["Investor_Name"].astype(str).str.replace(r"[\r\n]+", " ", regex=True).str.strip()
            )
        else:
            raise ValueError("CSV must include 'Investor_Name' column")

    # ---------- TEXT: fingerprint + TF-IDF ----------
    def _build_text_fingerprints(self):
        investors = self.df["Investor_Name"].unique()
        fingerprints = {}
        for inv in investors:
            rows = self.df[self.df["Investor_Name"] == inv]
            parts = []
            for col in self.text_cols:
                # join unique values into a text block
                parts.append(" ".join(rows[col].dropna().astype(str).unique()))
            fp = " ".join(parts).strip()
            fingerprints[inv] = fp
            self.investor_text_fingerprints[inv] = fp
        # fit vectorizer on all fingerprints
        self.vectorizer.fit(fingerprints.values())
        for inv in investors:
            vec = self.vectorizer.transform([fingerprints[inv]]).toarray()[0]
            self.investor_text_vectors[inv] = vec

    # ---------- NUMERIC: preprocessor + investor profiles ----------
    def _build_numeric_profiles(self):
        # numeric features + missingness flags
        numeric = self.numeric_cols.copy()
        # fill missing numeric values with median temporarily for fitting
        df_num = self.df[numeric].copy()
        # create missingness flags
        for col in numeric:
            self.df[f"{col}_missing"] = self.df[col].isna().astype(int)
        numeric_with_flags = numeric + [f"{c}_missing" for c in numeric]

        # For robust pipeline, fill numeric NaNs with medians for transformation
        medians = df_num.median()
        df_filled = df_num.fillna(medians)

        # build ColumnTransformer
        # Standard scale numeric_with_flags; onehot encode categorical
        # Ensure OneHotEncoder returns dense array
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        num_transform = ("num", StandardScaler(), numeric_with_flags)
        cat_transform = ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), self.cat_cols) if self.cat_cols else None
        transformers = [num_transform]
        if cat_transform:
            transformers.append(cat_transform)
        self.preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)

        # Fit preprocessor on rows (we need DataFrame with numeric_with_flags and cat cols)
        full_for_fit = self.df[numeric_with_flags + self.cat_cols].copy()
        # fill missing numeric values with medians (again)
        for col in numeric:
            full_for_fit[col] = full_for_fit[col].fillna(medians[col])
        # fit
        self.preprocessor.fit(full_for_fit)

        # Build investor-level numeric profiles by averaging per investor
        transformed_rows = []
        investor_names = []
        for idx, row in self.df.iterrows():
            # prepare row dict for transform
            r = {}
            for col in numeric:
                r[col] = row[col] if not pd.isna(row[col]) else medians[col]
                r[f"{col}_missing"] = int(pd.isna(row[col]))
            for col in self.cat_cols:
                r[col] = row[col]
            df_row = pd.DataFrame([r])
            arr = self.preprocessor.transform(df_row)
            transformed_rows.append(arr[0])
            investor_names.append(row["Investor_Name"])
        X = np.vstack(transformed_rows)
        X_df = pd.DataFrame(X, index=investor_names)
        # group by investor and average -> investor profiles
        self.investor_numeric_profiles = X_df.groupby(X_df.index).mean()

    # ---------- public generate method ----------
    def generate_profiles(self):
        """
        Call this to populate:
         - self.investor_text_vectors (dict inv -> np.array)
         - self.investor_numeric_profiles (DataFrame indexed by investor)
        """
        self._build_text_fingerprints()
        self._build_numeric_profiles()

    # ---------- helper getters ----------
    def get_investor_list(self) -> List[str]:
        return list(self.investor_text_vectors.keys())

    def get_text_matrix(self) -> np.ndarray:
        # returns matrix shape (n_investors, n_text_features)
        invs = self.get_investor_list()
        return np.vstack([self.investor_text_vectors[i] for i in invs])

    def get_numeric_matrix(self) -> np.ndarray:
        # investor_numeric_profiles rows are indexed by investor name
        return self.investor_numeric_profiles.values

    def get_investor_context(self, inv: str) -> str:
        return self.investor_text_fingerprints.get(inv, "")


# -------------------------
# Hybrid Matching Graph
# -------------------------
class SuperInvestorMatcher:
    def __init__(self, pipeline: HybridDataPipeline, openrouter_key: str, alpha: float = 0.7, model: str = "anthropic/claude-3.5-sonnet"):
        """
        alpha: weight for text similarity in fusion (0..1). numeric weight = 1 - alpha.
        """
        self.data = pipeline
        self.key = openrouter_key
        self.alpha = float(alpha)
        self.model = model

        # precompute matrices
        self.investors = self.data.get_investor_list()
        self.text_matrix = self.data.get_text_matrix()  # shape (n, d_text)
        self.numeric_matrix = self.data.get_numeric_matrix()  # shape (n, d_num)

    # --- utilities ---
    def _cosine_sim_vec(self, vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
        # vec: shape (d,) or (1,d), mat: (n,d)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        sims = cosine_similarity(vec, mat)[0]
        # clamp or fill nans
        sims = np.nan_to_num(sims)
        return sims

    # --- retrieval: compute separate similarities and fused score ---
    def retrieve_candidates(self, startup_profile: Dict, top_k: int = 10) -> List[Dict]:
        # build startup text vector
        startup_text = f"{startup_profile.get('industry','')} {startup_profile.get('description','')}"
        startup_text_vec = self.data.vectorizer.transform([startup_text]).toarray()[0]
        text_sims = self._cosine_sim_vec(startup_text_vec, self.text_matrix) * 100.0  # percentage

        # build startup numeric vector: need to construct same features used in preprocessor
        # numeric cols and missing flags
        numeric_cols = self.data.numeric_cols
        # prepare single-row dict for transformation
        r = {}
        medians = {}
        for col in numeric_cols:
            medians[col] = self.data.df[col].median() if col in self.data.df.columns else 0.0
            val = startup_profile.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                r[col] = medians[col]
                r[f"{col}_missing"] = 1
            else:
                r[col] = val
                r[f"{col}_missing"] = 0
        # include categorical
        for c in self.data.cat_cols:
            r[c] = startup_profile.get(c.lower(), startup_profile.get(c, ""))

        # transform using pipeline.preprocessor
        # if preprocessor expects list of cat cols present, build DataFrame accordingly
        row_df = pd.DataFrame([r])
        try:
            startup_num_vec = self.data.preprocessor.transform(row_df)
            if hasattr(startup_num_vec, "toarray"):
                startup_num_vec = startup_num_vec.toarray()[0]
            else:
                startup_num_vec = np.asarray(startup_num_vec)[0]
        except Exception as e:
            # fallback: zeros
            startup_num_vec = np.zeros(self.numeric_matrix.shape[1])

        numeric_sims = self._cosine_sim_vec(startup_num_vec, self.numeric_matrix) * 100.0

        # fusion
        alpha = self.alpha
        combined = alpha * text_sims + (1.0 - alpha) * numeric_sims

        # build results DataFrame-like list aligned with self.investors order
        results = []
        for i, inv in enumerate(self.investors):
            results.append({
                "investor": inv,
                "text_score": round(float(text_sims[i]), 2),
                "numeric_score": round(float(numeric_sims[i]), 2),
                "combined_score": round(float(combined[i]), 2)
            })

        # sort by combined_score desc
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_k]

    # --- web enrichment helper ---
    def fetch_investor_web(self, investor_name: str, progress_cb: ProgressCallback = None) -> str:
        if progress_cb:
            progress_cb(f"Fetching web summary for {investor_name}…")
        sys_msg = "You are a concise web researcher. Return 2 short bullet points describing this investor's focus and typical stage/check size. Keep each bullet <= 10 words."
        user_msg = f"Investor: {investor_name}"
        try:
            out = or_chat(self.model, [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}], self.key, max_tokens=120)
            return out.strip()
        except Exception as e:
            return f"(web unavailable: {e})"

    # --- LLM reasoning for a candidate ---
    def reason_about_candidate(self, startup: Dict, candidate: Dict, dataset_context: str, web_context: str, progress_cb: ProgressCallback = None) -> Dict:
        if progress_cb:
            progress_cb(f"Reasoning about {candidate['investor']}…")

        sys_prompt = (
            "You are a concise venture analyst. Using the dataset context, web context, embedding and numeric similarity scores, "
            "write one paragraph (max 4 sentences) explaining why this investor is a good or poor fit. "
            "Mention stage, deal-size alignment, sector fit, and any notable constraints. Be specific and evidence-based."
        )

        user_prompt = (
            f"STARTUP:\nIndustry: {startup.get('industry')}\nDeal Size: ${startup.get('Deal_Size_M', startup.get('deal_size_m', 'N/A'))}M\n"
            f"Growth YoY: {startup.get('Revenue_Growth_YoY', startup.get('revenue_growth_yoy', 'N/A'))}\n"
            f"Description: {startup.get('description')}\n\n"
            f"SCORES:\nText: {candidate['text_score']}\nNumeric: {candidate['numeric_score']}\nCombined: {candidate['combined_score']}\n\n"
            f"INVESTOR DATASET_CONTEXT:\n{dataset_context}\n\nWEB_SUMMARY:\n{web_context}\n\n"
            "Provide a concise, persuasive paragraph (<=4 sentences)."
        )

        try:
            explanation = or_chat(self.model, [{"role":"system","content":sys_prompt}, {"role":"user","content":user_prompt}], self.key, max_tokens=220)
            return {
                "investor": candidate["investor"],
                "text_score": candidate["text_score"],
                "numeric_score": candidate["numeric_score"],
                "combined_score": candidate["combined_score"],
                "explanation": explanation.strip(),
                "web": web_context
            }
        except Exception as e:
            return {
                "investor": candidate["investor"],
                "text_score": candidate["text_score"],
                "numeric_score": candidate["numeric_score"],
                "combined_score": candidate["combined_score"],
                "explanation": f"(LLM error: {e})",
                "web": web_context
            }

    # --- full pipeline run (with progress callback) ---
    def run_pipeline(self, startup_profile: Dict, top_k: int = 5, progress_cb: ProgressCallback = None) -> List[Dict]:
        if progress_cb:
            progress_cb("Computing similarities (text + numeric)…")
        candidates = self.retrieve_candidates(startup_profile, top_k=top_k)

        # web + reasoning for each candidate
        results = []
        for i, cand in enumerate(candidates, start=1):
            web = self.fetch_investor_web(cand["investor"], progress_cb=progress_cb)
            dataset_ctx = self.data.get_investor_context(cand["investor"])
            res = self.reason_about_candidate(startup_profile, cand, dataset_context=dataset_ctx, web_context=web, progress_cb=progress_cb)
            results.append(res)
        # final sort by combined_score (should already be sorted)
        results = sorted(results, key=lambda x: x["combined_score"], reverse=True)
        if progress_cb:
            progress_cb("Done.")
        return results
