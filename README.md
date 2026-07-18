# AI Traveller 🗺️

🚀 **[Live Demo](https://agentic-ai-traveller.streamlit.app)** (Replace this with your deployed Streamlit URL once completed!)

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2.9-orange.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Hariom312003/Agentic-AI-Traveller/pulls)

AI Traveller is an enterprise-grade **Agentic AI Travel Planner** built using a multi-agent orchestration pattern on top of **LangGraph**. The system leverages **Dynamic RAG (Retrieval-Augmented Generation)**, **Stateful Memory**, **Circuit-Breaker Protected Multi-LLM Routing**, and a **Self-Evaluation Critic Loop** to construct highly personalized, geographically optimized, and budget-aligned travel itineraries.

---

## 🌍 Production Deployment Guide

Deploying the system to production involves deploying the **FastAPI Backend** (on **Render**) and the **Streamlit Frontend** (on **Streamlit Community Cloud**).

### 1. Deploy the Backend on Render
Render reads the `render.yaml` file in this repository to configure the environment automatically.

1. Create a free account on **[Render.com](https://render.com/)**.
2. Click **New +** in the dashboard and select **Blueprint**.
3. Connect your GitHub repository: `Hariom312003/Agentic-AI-Traveller`.
4. Render will automatically detect the configuration from `render.yaml`:
   * **Name**: `agentic-ai-traveller-backend`
   * **Environment**: `Python 3`
   * **Build Command**: `pip install -r requirements.txt && python scripts/ingest_data.py` (installs libraries and builds the seed database).
   * **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
5. In the Render environment configuration, enter your **NEW Gemini API Key** under the key `GEMINI_API_KEY`.
6. Click **Deploy**. Copy your deployed service URL once the build succeeds (e.g. `https://agentic-ai-traveller-backend.onrender.com`).

---

### 2. Deploy the Frontend on Streamlit Community Cloud

1. Log in to **[Streamlit Community Cloud](https://share.streamlit.io/)**.
2. Click **New app** and connect your GitHub repository `Hariom312003/Agentic-AI-Traveller`.
3. Configure the app details:
   * **Main file path**: `frontend/main.py`
   * **Branch**: `main`
4. Expand the **Advanced settings** section at the bottom.
5. In the **Secrets** textbox, define the API URL to connect the frontend to your Render backend:
   ```toml
   API_BASE_URL = "https://your-deployed-render-url.onrender.com"
   ```
6. Click **Deploy**.

---

## 🗺️ Project Architecture Overview

```mermaid
graph TD
    User([User Prompt]) --> QueryAgent[Query Agent: Input Safety & Extraction]
    
    subgraph Multi-Agent Graph
        QueryAgent --> MemoryAgent[Memory Agent: Personalization Retrieval]
        MemoryAgent --> RAGAgent[RAG Agent: Worldwide Knowledge Acquisition]
        RAGAgent --> PlannerAgent[Planner Agent: Geographic & Theme Optimization]
        PlannerAgent --> BudgetAgent[Budget Agent: Local Estimator & Costing]
        BudgetAgent --> RewardsAgent[Rewards Agent: Illustrative Savings Registry]
        RewardsAgent --> ValidatorAgent[Validator Agent: Hard Constraints Audit]
        
        ValidatorAgent -- Replan Needed --> PlannerAgent
        ValidatorAgent -- Constraints Met --> EvaluatorAgent[Evaluator Agent: Self-Critic Loop]
        
        EvaluatorAgent -- Score < 8.5 & Attempts < 3 --> PlannerAgent
        EvaluatorAgent -- Score >= 8.5 --> MemoryUpdate[Memory Update Agent: Preference Persistence]
    end
    
    MemoryUpdate --> SummaryAgent[Summary Agent: Conversational Narrative]
    SummaryAgent --> UI[Streamlit Interactive Dashboard]
    
    subgraph Dynamic RAG Engine
        RAGAgent --> WikiRetriever[Wikivoyage / Wikipedia Scraping API]
        WikiRetriever --> VectorStore[(ChromaDB Vector Store)]
        WikiRetriever --> LexicalStore[(BM25 Lexical Store)]
    end
```

---

## 🌟 Key Features

### 1. Multi-Agent Orchestration & Workflow
- **LangGraph Workflow**: Programmed as a stateful compiled graph with explicit checkpointing.
- **Self-Evaluation Critic**: A critic loop that evaluates each generated itinerary's diversity, routing efficiency, user preference alignment, and budget compliance on a scale of `0.0 - 10.0`. It automatically replans up to 3 times if the score is below `8.5`.
- **Constraint Validator**: Audits slot overlaps, coordinate consistency, and duplicate attractions using RapidFuzz matching.

### 2. Intelligent Dynamic RAG & Knowledge Acquisition
- **On-the-Fly Scraping**: Automatically checks local database caches. For uncached destinations, it executes web extraction across Wikivoyage and Wikipedia API endpoints.
- **Hybrid Retrieval**: Combines sparse lexical matching (BM25) with dense vector embeddings (ChromaDB) to ground planners in real local attractions.
- **Centroid-Based Route Optimization**: Replaces missing attraction coordinates with clustered daily centroid coordinate averages, optimizing the Nearest Neighbor Traveling Salesperson (TSP) routing.

### 3. Concurrency Resilience & Safety Shields
- **Multi-LLM Failover Registry**: Implements a thread-safe registry that routes from Gemini (Primary) -> Groq -> OpenRouter -> Claude -> OpenAI -> Offline Fallback Planner.
- **Circuit Breaker Registry**: Temporarily blocks failing or rate-limited APIs (e.g. 429 RESOURCE_EXHAUSTED) with a cooldown period, gracefully degrading execution to rule-based fallback planners instead of crashing.
- **Input Safety Filter**: Protects the system against prompt injections (e.g. `Ignore previous instructions`) and symbol/emoji-only queries (e.g. `🏖️🏖️🏖️🏖️`).

---

## 📁 Repository Structure

```
Agentic-AI-Traveller/
├── data/                      # Curated destination seed files & JSON models
├── frontend/                  # Streamlit application layout and views
│   ├── api_client.py          # API wrappers for FastAPI endpoints
│   ├── main.py                # Streamlit entry point
│   └── views/                 # View tabs (trip planner, log monitor, memory dashboard)
├── src/                       # Backend Source Code
│   ├── agents/                # LangGraph Node Agents (Planner, Critic, Safety, etc.)
│   ├── api/                   # FastAPI routes, schemas, and app lifecycles
│   ├── graph/                 # LangGraph workflow configurations and checkpointers
│   ├── llm/                   # Multi-LLM Routing and Circuit Breakers
│   ├── models/                # Pydantic schemas and state definitions
│   ├── planning_engine/       # Route optimization, K-Means clustering, and TSP solvers
│   ├── rag/                   # Embeddings, chunking, and Wikipedia retrievers
│   └── validation/            # Hard constraints and input safety validators
├── render.yaml                # Render Blueprint deployment configuration
├── tests/                     # Unit and Integration test suite
├── Dockerfile                 # Production Docker build
├── docker-compose.yml         # Local container configurations
├── run.sh                     # Unified local start helper script
├── requirements.txt           # Python pinned dependencies
└── LICENSE                    # MIT License
```

---

## ⚙️ Installation & Setup (Local)

### 1. Clone the Repository
```bash
git clone https://github.com/Hariom312003/Agentic-AI-Traveller.git
cd Agentic-AI-Traveller
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Provide your API keys inside `.env`:
```ini
GEMINI_API_KEY=your_gemini_key_here
```

### 3. Fast Startup
```bash
chmod +x run.sh
./run.sh
```

---

## 🧪 Testing

```bash
pytest
```

---

## 📜 License
MIT License - see [LICENSE](LICENSE) for details.
