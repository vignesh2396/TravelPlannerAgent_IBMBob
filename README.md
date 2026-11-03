# ✈️ AI Travel Planning — Multi-Agent System

> **Domain:** Travel, Tourism & Hospitality  
> **LLM:** Google Gemini 2.5 Flash  
> **Powered by:** LangGraph · Streamlit · FastAPI · Python 3.11+  
> **Developed with:** IBM Bob (IBM SkillsBuild 2026 · Gartner Program · Day 02)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-purple)
![Tests](https://img.shields.io/badge/Tests-68%2F68%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

An intelligent travel planning application where **three specialized AI agents** collaborate to recommend **flights**, **hotels**, and **rental cars** — all coordinated through a single request with your destination, travel dates, total budget, and preferences.

> ⚠️ **All options displayed are SIMULATED (mock data)** for demonstration purposes.  
> No real bookings or payments are made without explicit user approval.

---

## 🏗️ Architecture

```
User (Streamlit UI)
       │  fills form: origin · destination · dates · budget · passengers
       ▼
FastAPI Backend  (/plan  ·  /approve  ·  /health)
       │
       ▼
LangGraph Orchestrator  (StateGraph — 6 nodes)
  │
  ├─ 1. collect_requirements   validates input · splits budget 40/45/15%
  ├─ 2. flight_agent     ──►  Google Gemini 2.5 Flash
  ├─ 3. hotel_agent      ──►  Google Gemini 2.5 Flash
  ├─ 4. car_agent        ──►  Google Gemini 2.5 Flash  (pickup time ← Agent 2 arrival)
  ├─ 5. validate_itinerary    pure Python: cost check · budget warning
  └─ 6. await_approval        itinerary_ready = True  →  UI shows results

Mock Data Layer  (MOCK_FLIGHTS · MOCK_HOTELS · MOCK_CARS — no paid APIs needed)
```

> The orchestrator is **workflow logic only** — it is NOT a fourth AI agent.

---

## 🤖 The Three AI Agents

| Agent | Input | Output | Prompt Focus |
|-------|-------|--------|-------------|
| **Agent 1 — Flight** | Origin, destination, dates, flight budget (40%) | ≤3 ranked flights with fare, class, stops, timing | Price / duration / stop balance |
| **Agent 2 — Hotel** | Destination, check-in/out, per-night budget (45%÷nights), preferences | ≤3 ranked hotels with rate, stars, amenities | Value · location · rating |
| **Agent 3 — Car** | Airport, pickup time ← Agent 1 arrival, passengers, daily budget (15%÷nights) | ≤3 ranked vehicles with daily rate, category | Passenger count · timing · cost |

**Budget split:** 40% flights · 45% hotel · 15% car rental

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| AI Agents | `google-genai >= 2.0.0` — Gemini 2.5 Flash, `temperature=0.3`, `max_tokens=1024` |
| Workflow | `langgraph >= 0.2.0` — `StateGraph` + `TypedDict` `TravelState` |
| Web UI | `streamlit >= 1.40.0` — tabbed results, sidebar form, custom CSS |
| REST API | `fastapi >= 0.115.0` + `uvicorn` + `pydantic >= 2.0.0` |
| Language | Python 3.11+ |
| Testing | `pytest >= 8.0.0` — 68/68 tests, 18 LLM dimensions, LLM mocked offline |
| Data | `mock_data.py` — 5 flights · 5 hotels · 5 cars (no paid travel APIs) |
| Video | `moviepy`, `Pillow`, `pyttsx3` — narrated slideshow generator |

---

## 📁 Project Structure

```
TravelPlannerAgent_IBMBob/
├── app.py                          # Streamlit UI (sidebar, tabs, cost summary, approval)
├── backend.py                      # FastAPI backend (/plan · /approve · /health)
├── workflow.py                     # LangGraph 6-node StateGraph + TravelState TypedDict
├── utils.py                        # Pure Python helpers (dates · budget · formatting)
├── generate_video.py               # Narrated MP4 slideshow generator (moviepy + PIL)
│
├── agents/
│   ├── __init__.py
│   └── agents.py                   # 3 AI agents — shared Gemini client, retry logic
│
├── data/
│   ├── __init__.py
│   └── mock_data.py                # MOCK_FLIGHTS · MOCK_HOTELS · MOCK_CARS + filter fns
│
├── tests/
│   ├── __init__.py
│   ├── test_travel_planner.py      # Original test suite (13 tests)
│   └── test_llm_agent_correctness.py  # LLM correctness suite (68 tests, 18 dimensions)
│
├── AI_Travel_Planner_MAS.pptx      # 12-slide project presentation
├── requirements.txt
├── .env.example                    # Template — copy to .env and add GEMINI_API_KEY
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/vignesh2396/TravelPlannerAgent_IBMBob.git
cd TravelPlannerAgent_IBMBob
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Gemini API key

```bash
# Copy the template
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux

# Edit .env — add your key:
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Get a **free** key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🚀 Running Options

### Option A — Streamlit UI only (recommended for demos)

```bash
streamlit run app.py
```

### Option B — FastAPI backend + Streamlit UI

**Terminal 1 — backend:**
```bash
uvicorn backend:app --reload --port 8000
```
Swagger UI available at [http://localhost:8000/docs](http://localhost:8000/docs)

**Terminal 2 — frontend:**
```bash
streamlit run app.py
```

---

## 🧪 Running Tests

```bash
# Original test suite (13 tests)
pytest tests/test_travel_planner.py -v

# LLM correctness suite (68 tests, 18 dimensions)
pytest tests/test_llm_agent_correctness.py -v --tb=short

# All tests together
pytest tests/ -v
```

All LLM calls are **mocked** — tests run fully offline with no API key required.

### Test Coverage — 18 LLM Testing Dimensions

| # | Dimension | Tests |
|---|-----------|-------|
| 1 | Accuracy | 6 |
| 2 | Completeness | 4 |
| 3 | Format Fidelity | 2 |
| 4 | Budget Adherence | 3 |
| 5 | Input Boundary | 6 |
| 6 | Negative Cases | 7 |
| 7 | Robustness (prompt injection, unicode) | 3 |
| 8 | Multi-Passenger Scaling | 2 |
| 9 | Data Types | 3 |
| 10 | Determinism | 1 |
| 11 | Agent Isolation | 2 |
| 12 | Workflow Integration | 3 |
| 13 | Date Edge Cases | 5 |
| 14 | Cost Calculation Arithmetic | 5 |
| 15 | Budget Split | 3 |
| 16 | Mock Data Filters | 6 |
| 17 | No-Recommendation Handling | 4 |
| 18 | Response Parsing | 3 |
| | **Total** | **68 / 68 ✅** |

---

## 🌐 REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check + model info |
| `POST` | `/plan` | Run all three agents, return itinerary + session_id |
| `POST` | `/approve` | Record approval, return simulated confirmation |

### POST /plan — Example Request

```json
{
  "origin": "New York (JFK)",
  "destination": "Los Angeles (LAX)",
  "departure_date": "2025-06-01",
  "return_date": "2025-06-07",
  "total_budget_usd": 150000,
  "hotel_preferences": "pool, free WiFi",
  "passengers": 2
}
```

---

## 🎬 Generating the Presentation Video

```bash
# Install extra dependencies (already in requirements.txt)
pip install moviepy pillow pyttsx3

# Render narrated 8-minute MP4 slideshow
python generate_video.py
```

Output: `AI_Travel_Planner_MAS.mp4` (~13 MB, 720p, with offline narration)

---

## 📊 Mock Data Summary

| Type | Count | Price Range |
|------|-------|-------------|
| Flights (JFK → LAX) | 5 | ₹13,500 – ₹57,000 |
| Hotels (Los Angeles) | 5 | ₹5,900 – ₹27,000 / night |
| Rental Cars (LAX) | 5 | ₹3,200 – ₹14,700 / day |

---

## ⚠️ Important Disclaimers

- All flight, hotel and car options are **SIMULATED mock data**
- No real travel provider APIs are called
- No payments or real bookings are processed at any point
- The Gemini API free tier has rate limits — agents include exponential back-off retry
- **Never** commit your `.env` file or API key to version control

---

## 🔮 Possible Next Steps

1. Integrate real travel APIs (Amadeus, Booking.com, car rental aggregators)
2. Add a Supervisor meta-agent to dynamically re-route on budget overage
3. Persist sessions in PostgreSQL / Redis instead of in-memory dict
4. Add an Activity / Itinerary Agent (Agent 4) for sightseeing recommendations
5. Deploy to cloud (Streamlit Community Cloud / Railway / AWS Lambda)
6. Add streaming LLM output to the UI using `st.write_stream`

---

## 👨‍💻 Development

This project was developed using **IBM Bob** as the AI coding assistant:

- **Plan mode** — designed project structure, agent responsibilities, data models and LangGraph workflow
- **Agent mode** — generated Streamlit UI, LangGraph orchestrator, three agents, mock data, validation, unit tests, PowerPoint presentation, and narrated video

IBM Bob is a development tool only — it is not a runtime dependency.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
