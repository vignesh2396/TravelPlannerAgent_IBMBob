"""
FastAPI backend for the Travel Planning Multi-Agent System.

Endpoints:
  POST /plan          — kick off the full multi-agent workflow
  POST /approve       — record user approval (simulated booking)
  GET  /health        — health check

Run with:
    uvicorn backend:app --reload --port 8000
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from workflow import run_travel_planning

load_dotenv()

app = FastAPI(
    title="AI Travel Planning Multi-Agent System",
    description=(
        "Three specialized AI agents (Flight, Hotel, Car) powered by Google Gemini "
        "collaborate to build a coordinated, budget-aware travel itinerary. "
        "All options are SIMULATED — no real bookings are made without user approval."
    ),
    version="1.0.0",
)

# Allow the Streamlit frontend on localhost during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store (replace with a DB for production)
# ---------------------------------------------------------------------------
_sessions: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class PlanRequest(BaseModel):
    origin: str = Field(..., example="New York (JFK)")
    destination: str = Field(..., example="Los Angeles (LAX)")
    departure_date: str = Field(..., example="2025-06-01", description="yyyy-mm-dd")
    return_date: str = Field(..., example="2025-06-07", description="yyyy-mm-dd")
    total_budget_usd: float = Field(..., gt=0, example=150000)
    hotel_preferences: Optional[str] = Field(default="", example="pool, free WiFi")
    passengers: int = Field(default=1, ge=1, le=20)

    @field_validator("departure_date", "return_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("Date must be in yyyy-mm-dd format") from exc
        return v


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool
    selected_flight_id: Optional[str] = None
    selected_hotel_id: Optional[str] = None
    selected_car_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    timestamp: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Verify the service is running and report the configured Gemini model."""
    return HealthResponse(
        status="ok",
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/plan", tags=["Planning"])
def plan_trip(request: PlanRequest):
    """
    Invoke the three AI agents in sequence and return a coordinated itinerary.

    The response includes:
    - flight_result, hotel_result, car_result from the respective agents
    - validation_errors / validation_warnings from deterministic checks
    - estimated_total_cost_usd
    - session_id for follow-up approval

    ⚠️ All options are SIMULATED. No real booking is made here.
    """
    try:
        state = run_travel_planning(request.model_dump())
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {exc}",
        ) from exc

    if state.get("error") and not state.get("itinerary_ready"):
        raise HTTPException(status_code=422, detail=state["error"])

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {"state": state, "approved": False}

    return {
        "session_id": session_id,
        "itinerary_ready": state.get("itinerary_ready", False),
        "validation_errors": state.get("validation_errors", []),
        "validation_warnings": state.get("validation_warnings", []),
        "estimated_total_cost_usd": state.get("estimated_total_cost_usd", 0),
        "flight_result": state.get("flight_result", {}),
        "hotel_result": state.get("hotel_result", {}),
        "car_result": state.get("car_result", {}),
        "disclaimer": (
            "⚠️ All options shown are SIMULATED for demonstration purposes. "
            "No real bookings have been made. Approve to simulate a booking confirmation."
        ),
    }


@app.post("/approve", tags=["Booking"])
def approve_booking(request: ApprovalRequest):
    """
    Record user approval and return a simulated booking confirmation.

    ⚠️ This does NOT make a real booking. It simulates confirmation only.
    Real bookings require integration with provider APIs and explicit user consent.
    """
    session = _sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please re-plan.")

    if not request.approved:
        _sessions[request.session_id]["approved"] = False
        return {"message": "Booking cancelled. No charges applied.", "approved": False}

    state = session["state"]
    _sessions[request.session_id]["approved"] = True

    # Simulate confirmation numbers
    confirmations = {
        "flight_confirmation": f"FL-SIM-{request.selected_flight_id or 'AUTO'}",
        "hotel_confirmation": f"HT-SIM-{request.selected_hotel_id or 'AUTO'}",
        "car_confirmation": f"CAR-SIM-{request.selected_car_id or 'AUTO'}",
    }

    return {
        "approved": True,
        "message": (
            "✅ SIMULATED booking confirmed. "
            "In a production system this would trigger real provider API calls."
        ),
        "estimated_total_cost_usd": state.get("estimated_total_cost_usd", 0),
        "confirmations": confirmations,
        "disclaimer": (
            "⚠️ This is a simulated confirmation only. "
            "No real payment or booking has been processed."
        ),
    }


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
