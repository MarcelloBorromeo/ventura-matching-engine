<img width="1512" height="935" alt="IEC Ventura 3" src="https://github.com/user-attachments/assets/9a5c9d92-8ae3-4687-9722-3d3e0f653abb" />
<img width="1512" height="938" alt="IEC Ventura 4" src="https://github.com/user-attachments/assets/7b9945fe-0a21-4dbf-b5b4-be3985555c8d" />

# Ventura: MVP Investor Matching Engine

**Ventura** is the Minimum Viable Product (MVP) engineered for **International Elite Capital (IEC)** to validate an AI-driven approach to venture capital deal flow. This engine was built to replace manual, high-cost intermediation with a scalable, automated system capable of vetting startups against institutional investment theses.

## Project Context

International Elite Capital required a "technical moat" to differentiate their services in the crowded capital advisory market. The core objective of this MVP was to prove that an algorithm could outperform basic filtering by identifying high-signal matches based on nuance rather than just sector tags.

The system solves three specific problems:

1. **Inefficiency:** Replacing manual analyst review with automated initial vetting.
2. **Opacity:** Providing transparent, data-backed reasoning for why a startup matches an investor.
3. **Signal-to-Noise:** Ensuring investors only see deals that align with their historical IPO performance.

## Technical Architecture

The system is built on a modular "Pipeline-to-Graph" architecture, ensuring that data processing is decoupled from the decision-making logic.

### 1. Data Intelligence (InvestorDataPipeline)

The engine leverages a dataset of historical VC-backed IPOs.

* **Investor Fingerprinting:** The pipeline creates a unique text "fingerprint" for each investor by aggregating their portfolio companies, industry focuses, and business overviews.
* **Vectorization:** Using `TfidfVectorizer`, these fingerprints are transformed into high-dimensional vectors.
* **Similarity Scoring:** When a startup profile is entered, the engine calculates the Cosine Similarity between the startup's vector and the investor pool:

$$\text{score} = \frac{\mathbf{v}_{startup} \cdot \mathbf{v}_{investor}}{\|\mathbf{v}_{startup}\| \|\mathbf{v}_{investor}\|}$$

### 2. The Decision Graph (InvestorMatchingGraph)

Ventura uses a multi-stage graph process to move from raw data to actionable insights:

* **Retrieve:** Performs a vector search to find the top candidates based on historical alignment.
* **Reason:** This is the "AI Moat." The engine passes the top matches to Claude 3.5 Sonnet (via OpenRouter). It synthesizes the startup's metrics with the investor's historical data and real-time web summaries to write a specific thesis on why the match is viable.
* **Rank:** Finalizes the order of recommendations based on both quantitative similarity and qualitative LLM reasoning.

## MVP Workflow Design

To minimize friction during the validation phase, the MVP was designed around a "Four Pillar" input system. The engine requires only four specific data points to generate a valid thesis:

1. **Industry:** The primary sector (e.g., Biotechnology, SaaS).
2. **Deal Size:** The target capital raise.
3. **Growth Rate:** Year-over-Year revenue metrics.
4. **Description:** A natural language overview of the value proposition.

## Premium Interface

The application interface was built using Streamlit with a custom CSS injection to align with IEC’s brand identity (Montserrat/Open Sans typography). It features a "Live Pipeline" status indicator to visually demonstrate the processing stages—Retrieving, Analyzing, and Ranking—providing immediate visual feedback to the user during the matching process.
