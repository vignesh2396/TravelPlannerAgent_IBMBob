"""
LangGraph orchestrator for the Multi-Agent Travel Planning System.

Workflow:
  START
    → collect_requirements    (validate & normalise user input)
    → run_flight_agent        (Agent 1)
    → run_hotel_agent         (Agent 2)
    → run_car_agent           (Agent 3)
    → validate_itinerary      (deterministic checks: dates, budget, timing)
    → await_approval          (surface itinerary to user; stop here)
  END                         (triggered after user approves)

The orchestrator is workflow logic only — it is NOT a fourth AI agent.
Cost calculations, date validation and budget enforcement use plain Python.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agents.agents import run_car_agent, run_flight_agent, run_hotel_agent
from data.mock_data import get_cars, get_flights, get_hotels

# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------


class TravelState(TypedDict, total=False):
    # ── User inputs ──────────────────────────────────────────────────────────
    origin: str
    destination: str
    departure_date: str          # ISO yyyy-mm-dd
    return_date: str             # ISO yyyy-mm-dd
    total_budget_usd: float
    hotel_preferences: str
    passengers: int

    # ── Derived ──────────────────────────────────────────────────────────────
    num_nights: int
    num_days: int                # car rental days (= num_nights)
    flight_budget_usd: float
    hotel_budget_per_night_usd: float
    car_budget_per_day_usd: float

    # ── Agent outputs ────────────────────────────────────────────────────────
    flight_result: Dict[str, Any]
    hotel_result: Dict[str, Any]
    car_result: Dict[str, Any]

    # ── Validation & itinerary ───────────────────────────────────────────────
    validation_errors: List[str]
    validation_warnings: List[str]
    estimated_total_cost_usd: float
    itinerary_ready: bool
    approved: bool
    error: str


# ---------------------------------------------------------------------------
# Helper: parse ISO date
# ---------------------------------------------------------------------------


def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _nights_between(checkin: str, checkout: str) -> int:
    return (_parse_date(checkout) - _parse_date(checkin)).days


# ---------------------------------------------------------------------------
# Node 1 — collect_requirements
# ---------------------------------------------------------------------------


def collect_requirements(state: TravelState) -> TravelState:
    """
    Validate user inputs and split the total budget across the three agents.
    Uses deterministic Python logic — no LLM call.

    Budget split (approximate defaults):
      40% flights, 45% hotel, 15% car
    """
    errors: List[str] = []

    # Validate dates
    try:
        dep = _parse_date(state["departure_date"])
        ret = _parse_date(state["return_date"])
        if ret <= dep:
            errors.append("Return date must be after departure date.")
        num_nights = (ret - dep).days
    except (ValueError, KeyError):
        errors.append("Dates must be in yyyy-mm-dd format.")
        num_nights = 0

    # Validate budget
    budget = float(state.get("total_budget_usd", 0))
    if budget <= 0:
        errors.append("Total budget must be greater than 0.")

    passengers = int(state.get("passengers", 1))
    if passengers < 1:
        passengers = 1

    # Budget split
    flight_budget = round(budget * 0.40, 2)
    hotel_budget_per_night = round((budget * 0.45) / max(num_nights, 1), 2)
    car_budget_per_day = round((budget * 0.15) / max(num_nights, 1), 2)

    return {
        **state,
        "num_nights": num_nights,
        "num_days": num_nights,
        "flight_budget_usd": flight_budget,
        "hotel_budget_per_night_usd": hotel_budget_per_night,
        "car_budget_per_day_usd": car_budget_per_day,
        "validation_errors": errors,
        "validation_warnings": [],
        "itinerary_ready": False,
        "approved": False,
        "error": "" if not errors else "; ".join(errors),
    }


# ---------------------------------------------------------------------------
# Node 2 — flight_agent_node
# ---------------------------------------------------------------------------


def flight_agent_node(state: TravelState) -> TravelState:
    """Run Agent 1 (Flight Booking Agent)."""
    if state.get("error"):
        return state

    flights = get_flights(
        origin=state.get("origin", ""),
        destination=state.get("destination", ""),
        budget=state.get("flight_budget_usd", 0),
    )

    result = run_flight_agent(
        origin=state.get("origin", ""),
        destination=state.get("destination", ""),
        departure_date=state.get("departure_date", ""),
        return_date=state.get("return_date", ""),
        budget_usd=state.get("flight_budget_usd", 0),
        flights=flights,
    )

    return {**state, "flight_result": result}


# ---------------------------------------------------------------------------
# Node 3 — hotel_agent_node
# ---------------------------------------------------------------------------


def hotel_agent_node(state: TravelState) -> TravelState:
    """Run Agent 2 (Hotel Booking Agent)."""
    if state.get("error"):
        return state

    hotels = get_hotels(
        destination=state.get("destination", ""),
        budget_per_night=state.get("hotel_budget_per_night_usd", 0),
    )

    result = run_hotel_agent(
        destination=state.get("destination", ""),
        check_in=state.get("departure_date", ""),
        check_out=state.get("return_date", ""),
        num_nights=state.get("num_nights", 1),
        budget_per_night_usd=state.get("hotel_budget_per_night_usd", 0),
        preferences=state.get("hotel_preferences", ""),
        hotels=hotels,
    )

    return {**state, "hotel_result": result}


# ---------------------------------------------------------------------------
# Node 4 — car_agent_node
# ---------------------------------------------------------------------------


def car_agent_node(state: TravelState) -> TravelState:
    """Run Agent 3 (Car Booking Agent)."""
    if state.get("error"):
        return state

    cars = get_cars(daily_budget=state.get("car_budget_per_day_usd", 0))

    # Derive pickup time from flight recommendation if available
    flight_recs = (state.get("flight_result") or {}).get("recommendations", [])
    pickup_time = flight_recs[0]["arrival"] if flight_recs else "TBD"

    result = run_car_agent(
        pickup_location=f"{state.get('destination', '')} Airport",
        pickup_time=pickup_time,
        dropoff_time=f"Day {state.get('num_days', 1)} evening",
        num_days=state.get("num_days", 1),
        budget_per_day_usd=state.get("car_budget_per_day_usd", 0),
        passengers=state.get("passengers", 1),
        cars=cars,
    )

    return {**state, "car_result": result}


# ---------------------------------------------------------------------------
# Node 5 — validate_itinerary (deterministic)
# ---------------------------------------------------------------------------


def validate_itinerary(state: TravelState) -> TravelState:
    """
    Deterministic validation: check dates, timing dependencies and total cost.
    Updates validation_errors, validation_warnings and estimated_total_cost_usd.
    """
    if state.get("error"):
        return state

    errors: List[str] = list(state.get("validation_errors", []))
    warnings: List[str] = list(state.get("validation_warnings", []))
    total = 0.0

    # ── Flight cost ──────────────────────────────────────────────────────────
    flight_recs = (state.get("flight_result") or {}).get("recommendations", [])
    if flight_recs:
        best_flight_fare = float(flight_recs[0].get("fare_usd", 0))
        passengers = state.get("passengers", 1)
        total += best_flight_fare * passengers
    else:
        warnings.append("No flight recommendations were returned.")

    # ── Hotel cost ───────────────────────────────────────────────────────────
    hotel_recs = (state.get("hotel_result") or {}).get("recommendations", [])
    num_nights = state.get("num_nights", 0)
    if hotel_recs:
        best_hotel_rate = float(hotel_recs[0].get("nightly_rate_usd", 0))
        total += best_hotel_rate * num_nights
    else:
        warnings.append("No hotel recommendations were returned.")

    # ── Car cost ─────────────────────────────────────────────────────────────
    car_recs = (state.get("car_result") or {}).get("recommendations", [])
    num_days = state.get("num_days", 0)
    if car_recs:
        best_car_rate = float(car_recs[0].get("daily_rate_usd", 0))
        total += best_car_rate * num_days
    else:
        warnings.append("No car recommendations were returned.")

    # ── Budget check ─────────────────────────────────────────────────────────
    budget = float(state.get("total_budget_usd", 0))
    if total > budget:
        overage = total - budget
        warnings.append(
            f"Estimated total (INR {total:.0f}) exceeds budget by INR {overage:.0f}. "
            "Consider adjusting your preferences."
        )

    # ── Date check ───────────────────────────────────────────────────────────
    if num_nights <= 0:
        errors.append("Trip duration must be at least 1 night.")

    itinerary_ready = len(errors) == 0

    return {
        **state,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "estimated_total_cost_usd": round(total, 2),
        "itinerary_ready": itinerary_ready,
        "error": "; ".join(errors) if errors else "",
    }


# ---------------------------------------------------------------------------
# Conditional edge: stop for user approval or surface error
# ---------------------------------------------------------------------------


def should_continue(state: TravelState) -> str:
    if state.get("error"):
        return END
    return "await_approval"


# ---------------------------------------------------------------------------
# Node 6 — await_approval (terminal state; UI reads itinerary_ready flag)
# ---------------------------------------------------------------------------


def await_approval(state: TravelState) -> TravelState:
    """
    Mark the itinerary as ready for user review.
    The Streamlit UI presents the recommendations here and collects approval.
    No LLM call — this is a workflow coordination node only.
    """
    return {**state, "itinerary_ready": True}


# ---------------------------------------------------------------------------
# Build and compile the LangGraph workflow
# ---------------------------------------------------------------------------


def build_workflow() -> Any:
    graph = StateGraph(TravelState)

    graph.add_node("collect_requirements", collect_requirements)
    graph.add_node("flight_agent", flight_agent_node)
    graph.add_node("hotel_agent", hotel_agent_node)
    graph.add_node("car_agent", car_agent_node)
    graph.add_node("validate_itinerary", validate_itinerary)
    graph.add_node("await_approval", await_approval)

    graph.add_edge(START, "collect_requirements")
    graph.add_edge("collect_requirements", "flight_agent")
    graph.add_edge("flight_agent", "hotel_agent")
    graph.add_edge("hotel_agent", "car_agent")
    graph.add_edge("car_agent", "validate_itinerary")
    graph.add_conditional_edges("validate_itinerary", should_continue)
    graph.add_edge("await_approval", END)

    return graph.compile()


# Module-level compiled app
travel_workflow = build_workflow()


def run_travel_planning(user_inputs: Dict[str, Any]) -> TravelState:
    """
    Entry point for the orchestrator.

    Args:
        user_inputs: dict with keys origin, destination, departure_date,
                     return_date, total_budget_usd, hotel_preferences, passengers.

    Returns:
        Final TravelState with all agent results and validation.
    """
    initial_state: TravelState = {
        "origin": user_inputs.get("origin", ""),
        "destination": user_inputs.get("destination", ""),
        "departure_date": user_inputs.get("departure_date", ""),
        "return_date": user_inputs.get("return_date", ""),
        "total_budget_usd": float(user_inputs.get("total_budget_usd", 0)),
        "hotel_preferences": user_inputs.get("hotel_preferences", ""),
        "passengers": int(user_inputs.get("passengers", 1)),
    }
    final_state = travel_workflow.invoke(initial_state)
    return final_state
