"""
Three specialized AI travel agents powered by Google Gemini API.

Agent 1 — Flight Booking Agent
Agent 2 — Hotel Booking Agent
Agent 3 — Car Booking Agent

Each agent has a distinct system prompt, receives structured input,
and returns a structured recommendation. The Gemini client is shared
and configured through GEMINI_API_KEY and GEMINI_MODEL env variables.

NOTE: All recommendations are based on SIMULATED mock data.
      No real bookings are made without explicit user approval.
"""

import json
import os
import time
from typing import Any, Dict, List

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Shared Gemini client (initialised lazily on first use)
# ---------------------------------------------------------------------------
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Add it to your .env file and restart the application."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _get_model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


# ---------------------------------------------------------------------------
# Internal helper — call Gemini with retry on quota/rate-limit errors
# ---------------------------------------------------------------------------
def _extract_text(response: Any) -> str:
    """
    Safely extract text from a Gemini response object.
    response.text can be None for some models (e.g. gemini-2.5-flash);
    fall back to iterating candidates → content → parts.
    """
    # Fast path — works for most models
    if response.text is not None:
        return response.text

    # Fallback: walk candidates
    try:
        for candidate in response.candidates or []:
            for part in (candidate.content.parts or []):
                if part.text:
                    return part.text
    except Exception:
        pass
    return ""


def _call_gemini(system_prompt: str, user_message: str, retries: int = 3) -> str:
    """
    Send a request to Gemini and return the text response.
    Retries up to `retries` times on 429 / resource-exhausted / 503 errors.
    """
    client = _get_client()
    model = _get_model()

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            )
            return _extract_text(response)
        except Exception as exc:
            err_str = str(exc).lower()
            if "quota" in err_str or "rate" in err_str or "429" in err_str or "503" in err_str or "unavailable" in err_str:
                wait = 2 ** attempt  # exponential back-off: 1s, 2s, 4s
                time.sleep(wait)
                if attempt == retries - 1:
                    raise
            else:
                raise
    return ""


# ---------------------------------------------------------------------------
# Agent 1 — Flight Booking Agent
# ---------------------------------------------------------------------------
FLIGHT_AGENT_SYSTEM_PROMPT = """
You are an expert Flight Booking Agent for an AI travel planning system.
Your job is to analyse available flight options and recommend the best flights
based on the traveller's requirements.

Rules:
- Always clarify that these are SIMULATED options, not real bookings.
- Rank options by value (balance of price, duration and timing).
- Respect the user's budget — do not recommend options over budget unless
  no cheaper option exists, in which case note the shortfall clearly.
- Return a concise JSON object with key "recommendations" (a list of up to 3 flights).
- Each flight item must have: id, airline, departure, arrival, duration,
  fare_usd, class, stops, reason (why recommended).
- After the JSON, add a brief plain-English summary paragraph.
"""


def run_flight_agent(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str,
    budget_usd: float,
    flights: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Invoke the Flight Booking Agent.

    Returns a dict with:
      - "recommendations": list of ranked flight dicts
      - "summary": plain-English explanation
      - "raw_response": full LLM text (for debugging)
    """
    user_message = f"""
Travel requirements:
- Origin: {origin}
- Destination: {destination}
- Departure date: {departure_date}
- Return date: {return_date}
- Flight budget (one-way): INR {budget_usd}

Available (simulated) flight options:
{json.dumps(flights, indent=2)}

Please recommend the top 3 flights ranked by best value, respecting the budget.
Return JSON first, then a plain-English summary.
"""
    raw = _call_gemini(FLIGHT_AGENT_SYSTEM_PROMPT, user_message)

    recommendations: List[Dict] = []
    summary = raw

    # Try to extract the JSON block from the response
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        data = json.loads(raw[start:end])
        recommendations = data.get("recommendations", [])
        # Everything after the JSON is the summary
        summary = raw[end:].strip() or "See recommendations above."
    except (ValueError, json.JSONDecodeError):
        # LLM returned plain text — surface it as summary
        summary = raw

    return {
        "recommendations": recommendations,
        "summary": summary,
        "raw_response": raw,
    }


# ---------------------------------------------------------------------------
# Agent 2 — Hotel Booking Agent
# ---------------------------------------------------------------------------
HOTEL_AGENT_SYSTEM_PROMPT = """
You are an expert Hotel Booking Agent for an AI travel planning system.
Your job is to analyse available hotel options and recommend the best
accommodations based on the traveller's needs.

Rules:
- Always clarify that these are SIMULATED options, not real bookings.
- Coordinate check-in / check-out dates with the provided travel schedule.
- Recommend options that fit within the per-night budget where possible.
- Return a concise JSON object with key "recommendations" (a list of up to 3 hotels).
- Each hotel item must have: id, name, stars, nightly_rate_usd, total_cost_usd,
  location, rating, amenities, reason (why recommended).
- After the JSON, add a brief plain-English summary paragraph.
"""


def run_hotel_agent(
    destination: str,
    check_in: str,
    check_out: str,
    num_nights: int,
    budget_per_night_usd: float,
    preferences: str,
    hotels: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Invoke the Hotel Booking Agent.

    Returns a dict with:
      - "recommendations": list of ranked hotel dicts
      - "summary": plain-English explanation
      - "raw_response": full LLM text
    """
    # Attach computed total cost to each hotel for the LLM
    hotels_with_total = [
        {**h, "total_cost_usd": h["nightly_rate_usd"] * num_nights}
        for h in hotels
    ]

    user_message = f"""
Travel requirements:
- Destination: {destination}
- Check-in date: {check_in}
- Check-out date: {check_out}
- Number of nights: {num_nights}
- Budget per night: INR {budget_per_night_usd}
- Preferences: {preferences or 'None specified'}

Available (simulated) hotel options (total_cost_usd = nightly_rate × nights):
{json.dumps(hotels_with_total, indent=2)}

Please recommend the top 3 hotels ranked by best value, respecting the budget.
Return JSON first, then a plain-English summary.
"""
    raw = _call_gemini(HOTEL_AGENT_SYSTEM_PROMPT, user_message)

    recommendations: List[Dict] = []
    summary = raw

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        data = json.loads(raw[start:end])
        recommendations = data.get("recommendations", [])
        summary = raw[end:].strip() or "See recommendations above."
    except (ValueError, json.JSONDecodeError):
        summary = raw

    return {
        "recommendations": recommendations,
        "summary": summary,
        "raw_response": raw,
    }


# ---------------------------------------------------------------------------
# Agent 3 — Car Booking Agent
# ---------------------------------------------------------------------------
CAR_AGENT_SYSTEM_PROMPT = """
You are an expert Car Booking Agent for an AI travel planning system.
Your job is to recommend the most suitable rental car or transfer option
based on the traveller's arrival time, destination and budget.

Rules:
- Always clarify that these are SIMULATED options, not real bookings.
- Align pickup time with the flight arrival time provided.
- Recommend options within the daily budget where possible.
- Return a concise JSON object with key "recommendations" (a list of up to 3 vehicles).
- Each item must have: id, provider, vehicle, category, daily_rate_usd, total_cost_usd,
  pickup_location, passengers, reason (why recommended).
- After the JSON, add a brief plain-English summary paragraph.
"""


def run_car_agent(
    pickup_location: str,
    pickup_time: str,
    dropoff_time: str,
    num_days: int,
    budget_per_day_usd: float,
    passengers: int,
    cars: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Invoke the Car Booking Agent.

    Returns a dict with:
      - "recommendations": list of ranked car dicts
      - "summary": plain-English explanation
      - "raw_response": full LLM text
    """
    cars_with_total = [
        {**c, "total_cost_usd": c["daily_rate_usd"] * num_days}
        for c in cars
    ]

    user_message = f"""
Transport requirements:
- Pickup location: {pickup_location}
- Pickup time: {pickup_time}
- Drop-off time: {dropoff_time}
- Number of days: {num_days}
- Daily budget: INR {budget_per_day_usd}
- Number of passengers: {passengers}

Available (simulated) vehicle options (total_cost_inr = daily_rate × days):
{json.dumps(cars_with_total, indent=2)}

Please recommend the top 3 vehicles ranked by best value, respecting the budget
and passenger count.
Return JSON first, then a plain-English summary.
"""
    raw = _call_gemini(CAR_AGENT_SYSTEM_PROMPT, user_message)

    recommendations: List[Dict] = []
    summary = raw

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        data = json.loads(raw[start:end])
        recommendations = data.get("recommendations", [])
        summary = raw[end:].strip() or "See recommendations above."
    except (ValueError, json.JSONDecodeError):
        summary = raw

    return {
        "recommendations": recommendations,
        "summary": summary,
        "raw_response": raw,
    }
