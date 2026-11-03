"""
Utility helpers: date validation, budget checking, itinerary formatting.
All pure Python — no LLM calls.
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------


def parse_date(date_str: str) -> date:
    """Parse an ISO-format date string (yyyy-mm-dd) to a date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def nights_between(checkin: str, checkout: str) -> int:
    """Return the number of nights between two ISO date strings."""
    return (parse_date(checkout) - parse_date(checkin)).days


def validate_dates(departure: str, return_date: str) -> Tuple[bool, str]:
    """
    Return (is_valid, error_message).
    is_valid is True when departure < return_date.
    """
    try:
        dep = parse_date(departure)
        ret = parse_date(return_date)
    except ValueError:
        return False, "Dates must be in yyyy-mm-dd format."
    if ret <= dep:
        return False, "Return date must be after departure date."
    return True, ""


# ---------------------------------------------------------------------------
# Budget utilities
# ---------------------------------------------------------------------------


def split_budget(
    total: float,
    num_nights: int,
    flight_pct: float = 0.40,
    hotel_pct: float = 0.45,
    car_pct: float = 0.15,
) -> Dict[str, float]:
    """
    Split a total budget across flight, hotel and car allocations.

    Returns:
        {
          "flight_budget_usd": float,
          "hotel_budget_per_night_usd": float,
          "car_budget_per_day_usd": float,
        }
    """
    nights = max(num_nights, 1)
    return {
        "flight_budget_usd": round(total * flight_pct, 2),
        "hotel_budget_per_night_usd": round((total * hotel_pct) / nights, 2),
        "car_budget_per_day_usd": round((total * car_pct) / nights, 2),
    }


def calculate_estimated_total(
    flight_fare: float,
    passengers: int,
    nightly_rate: float,
    num_nights: int,
    daily_car_rate: float,
    num_days: int,
) -> float:
    """
    Calculate estimated trip cost from the top-ranked agent recommendations.
    """
    flight_cost = flight_fare * passengers
    hotel_cost = nightly_rate * num_nights
    car_cost = daily_car_rate * num_days
    return round(flight_cost + hotel_cost + car_cost, 2)


def check_budget(estimated_total: float, budget: float) -> Tuple[bool, float]:
    """
    Return (within_budget, overage_or_saving).
    overage_or_saving is positive when under budget, negative when over.
    """
    diff = budget - estimated_total
    return diff >= 0, round(diff, 2)


# ---------------------------------------------------------------------------
# Itinerary formatter (plain-text summary for README / terminal output)
# ---------------------------------------------------------------------------


def format_itinerary(state: Dict[str, Any]) -> str:
    """
    Produce a human-readable plain-text itinerary summary from the workflow state.
    """
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("  AI TRAVEL ITINERARY SUMMARY  (SIMULATED)")
    lines.append("=" * 60)

    lines.append(
        f"\nTrip: {state.get('origin', '?')} → {state.get('destination', '?')}"
    )
    lines.append(
        f"Dates: {state.get('departure_date', '?')} to {state.get('return_date', '?')} "
        f"({state.get('num_nights', 0)} nights)"
    )
    lines.append(f"Passengers: {state.get('passengers', 1)}")
    lines.append(f"Total Budget: ₹{state.get('total_budget_usd', 0):,.0f}")
    lines.append(f"Estimated Cost: ₹{state.get('estimated_total_cost_usd', 0):,.0f}")

    # Flights
    flight_recs = (state.get("flight_result") or {}).get("recommendations", [])
    lines.append("\n--- TOP FLIGHT RECOMMENDATION ---")
    if flight_recs:
        f = flight_recs[0]
        lines.append(
            f"  {f.get('airline', 'N/A')} · {f.get('departure', '?')} → {f.get('arrival', '?')} "
            f"· ₹{f.get('fare_usd', '?')} · {f.get('class', '')} · {f.get('stops', 0)} stop(s)"
        )
    else:
        lines.append("  No flight recommendation available.")

    # Hotels
    hotel_recs = (state.get("hotel_result") or {}).get("recommendations", [])
    lines.append("\n--- TOP HOTEL RECOMMENDATION ---")
    if hotel_recs:
        h = hotel_recs[0]
        lines.append(
            f"  {h.get('name', 'N/A')} ⭐{h.get('stars', '?')} · "
            f"{h.get('location', '?')} · ₹{h.get('nightly_rate_usd', '?')}/night"
        )
    else:
        lines.append("  No hotel recommendation available.")

    # Cars
    car_recs = (state.get("car_result") or {}).get("recommendations", [])
    lines.append("\n--- TOP CAR RECOMMENDATION ---")
    if car_recs:
        c = car_recs[0]
        lines.append(
            f"  {c.get('vehicle', 'N/A')} ({c.get('category', '')}) · "
            f"{c.get('provider', '?')} · ₹{c.get('daily_rate_usd', '?')}/day"
        )
    else:
        lines.append("  No car recommendation available.")

    # Warnings
    warnings = state.get("validation_warnings", [])
    if warnings:
        lines.append("\n--- WARNINGS ---")
        for w in warnings:
            lines.append(f"  ⚠  {w}")

    lines.append("\n" + "=" * 60)
    lines.append("  ⚠ All options shown are SIMULATED mock data.")
    lines.append("  No real bookings or payments have been made.")
    lines.append("=" * 60)

    return "\n".join(lines)
