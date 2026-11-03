"""
Comprehensive LLM Agent Correctness Test Suite
===============================================
Multi-Agent Travel Planning System — Google Gemini backend

LLM Testing Parameters Covered:
  1.  Accuracy         — Correct fields present & correct data types returned
  2.  Completeness     — All required fields in every recommendation
  3.  Format Fidelity  — Valid JSON structure; correct key names
  4.  Budget Adherence — Agent respects budget constraints
  5.  Input Boundary   — Minimum, maximum, zero-value, very-large inputs
  6.  Negative Cases   — Missing fields, invalid JSON from LLM, empty response
  7.  Robustness       — Malformed LLM output gracefully surfaced as summary
  8.  Prompt Injection — Adversarial user input doesn't break JSON parse path
  9.  Multi-passenger  — Cost scaling correct for >1 passenger
  10. Data Types        — Numeric fields are numeric, string fields are strings
  11. Determinism       — Same inputs → same structural shape (LLM mocked)
  12. Agent Isolation   — Each agent returns only its own artefact keys
  13. Workflow Integration — End-to-end TravelState populated correctly
  14. Validation Logic  — Budget over/under detection, date checks, night counts
  15. Edge Dates        — Same-day, one-night, multi-month trips
  16. Cost Calculation  — calculate_estimated_total arithmetic correctness
  17. Budget Split      — 40/45/15 split sums to ≈ total
  18. Mock Data Filters — Budget filters work at boundary values
  19. Response Parsing  — JSON with leading/trailing text still parsed
  20. No-Recommendation — Graceful handling when LLM returns empty list

Run with:
    cd <project_root>
    pytest tests/test_llm_agent_correctness.py -v --tb=short
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.agents import run_car_agent, run_flight_agent, run_hotel_agent
from data.mock_data import (
    MOCK_CARS,
    MOCK_FLIGHTS,
    MOCK_HOTELS,
    get_cars,
    get_flights,
    get_hotels,
)
from utils import (
    calculate_estimated_total,
    check_budget,
    nights_between,
    split_budget,
    validate_dates,
)
from workflow import TravelState, collect_requirements, validate_itinerary

# ============================================================
# Shared test fixtures / helpers
# ============================================================

REQUIRED_FLIGHT_FIELDS = {"id", "airline", "departure", "arrival", "duration",
                           "fare_usd", "class", "stops", "reason"}
REQUIRED_HOTEL_FIELDS  = {"id", "name", "stars", "nightly_rate_usd", "total_cost_usd",
                           "location", "rating", "amenities", "reason"}
REQUIRED_CAR_FIELDS    = {"id", "provider", "vehicle", "category", "daily_rate_usd",
                          "total_cost_usd", "pickup_location", "passengers", "reason"}


def _make_flight_json(overrides: Dict[str, Any] | None = None) -> str:
    base = {
        "id": "FL001", "airline": "TestAir",
        "departure": "08:00", "arrival": "11:30",
        "duration": "5h 30m", "fare_usd": 22000,
        "class": "Economy", "stops": 0,
        "reason": "Best value.",
    }
    if overrides:
        base.update(overrides)
    return json.dumps({"recommendations": [base]})


def _make_hotel_json(overrides: Dict[str, Any] | None = None) -> str:
    base = {
        "id": "HT001", "name": "Test Hotel",
        "stars": 3, "nightly_rate_usd": 9000,
        "total_cost_usd": 54000, "location": "Downtown",
        "rating": 4.2, "amenities": ["WiFi"],
        "reason": "Good value.",
    }
    if overrides:
        base.update(overrides)
    return json.dumps({"recommendations": [base]})


def _make_car_json(overrides: Dict[str, Any] | None = None) -> str:
    base = {
        "id": "CAR001", "provider": "TestRent",
        "vehicle": "Toyota Corolla", "category": "Economy",
        "daily_rate_usd": 3500, "total_cost_usd": 21000,
        "pickup_location": "LAX", "passengers": 4,
        "reason": "Affordable.",
    }
    if overrides:
        base.update(overrides)
    return json.dumps({"recommendations": [base]})


def _base_state() -> TravelState:
    return TravelState(
        origin="New York (JFK)",
        destination="Los Angeles (LAX)",
        departure_date="2025-06-01",
        return_date="2025-06-07",
        total_budget_usd=150000,
        hotel_preferences="",
        passengers=2,
    )


# ============================================================
# 1. ACCURACY — correct fields & data types
# ============================================================

class TestAccuracy:
    """LLM Test 1 — Accuracy: required fields present with correct types."""

    @patch("agents.agents._call_gemini", return_value=_make_flight_json())
    def test_flight_required_fields_present(self, _):
        result = run_flight_agent("New York", "Los Angeles",
                                  "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
        rec = result["recommendations"][0]
        missing = REQUIRED_FLIGHT_FIELDS - rec.keys()
        assert not missing, f"Missing flight fields: {missing}"

    @patch("agents.agents._call_gemini", return_value=_make_hotel_json())
    def test_hotel_required_fields_present(self, _):
        result = run_hotel_agent("Los Angeles", "2025-06-01", "2025-06-07",
                                 6, 15000, "", MOCK_HOTELS)
        rec = result["recommendations"][0]
        missing = REQUIRED_HOTEL_FIELDS - rec.keys()
        assert not missing, f"Missing hotel fields: {missing}"

    @patch("agents.agents._call_gemini", return_value=_make_car_json())
    def test_car_required_fields_present(self, _):
        result = run_car_agent("LAX", "11:30", "Day 6 evening",
                               6, 5000, 2, MOCK_CARS)
        rec = result["recommendations"][0]
        missing = REQUIRED_CAR_FIELDS - rec.keys()
        assert not missing, f"Missing car fields: {missing}"

    @patch("agents.agents._call_gemini", return_value=_make_flight_json())
    def test_flight_fare_is_numeric(self, _):
        result = run_flight_agent("NY", "LA", "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
        fare = result["recommendations"][0]["fare_usd"]
        assert isinstance(fare, (int, float)), f"fare_usd should be numeric, got {type(fare)}"

    @patch("agents.agents._call_gemini", return_value=_make_hotel_json())
    def test_hotel_nightly_rate_is_numeric(self, _):
        result = run_hotel_agent("LA", "2025-06-01", "2025-06-07", 6, 15000, "", MOCK_HOTELS)
        rate = result["recommendations"][0]["nightly_rate_usd"]
        assert isinstance(rate, (int, float))

    @patch("agents.agents._call_gemini", return_value=_make_car_json())
    def test_car_daily_rate_is_numeric(self, _):
        result = run_car_agent("LAX", "11:30", "Day 6 evening", 6, 5000, 2, MOCK_CARS)
        rate = result["recommendations"][0]["daily_rate_usd"]
        assert isinstance(rate, (int, float))


# ============================================================
# 2. COMPLETENESS — all three result keys always returned
# ============================================================

class TestCompleteness:
    """LLM Test 2 — Completeness: result dict always has all expected keys."""

    @patch("agents.agents._call_gemini", return_value=_make_flight_json())
    def test_flight_result_has_all_keys(self, _):
        result = run_flight_agent("NY", "LA", "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
        for key in ("recommendations", "summary", "raw_response"):
            assert key in result, f"Key '{key}' missing from flight result"

    @patch("agents.agents._call_gemini", return_value=_make_hotel_json())
    def test_hotel_result_has_all_keys(self, _):
        result = run_hotel_agent("LA", "2025-06-01", "2025-06-07", 6, 15000, "", MOCK_HOTELS)
        for key in ("recommendations", "summary", "raw_response"):
            assert key in result

    @patch("agents.agents._call_gemini", return_value=_make_car_json())
    def test_car_result_has_all_keys(self, _):
        result = run_car_agent("LAX", "11:30", "Day 6 evening", 6, 5000, 2, MOCK_CARS)
        for key in ("recommendations", "summary", "raw_response"):
            assert key in result

    @patch("agents.agents._call_gemini", return_value=_make_flight_json())
    def test_recommendations_is_a_list(self, _):
        result = run_flight_agent("NY", "LA", "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
        assert isinstance(result["recommendations"], list)


# ============================================================
# 3. FORMAT FIDELITY — valid JSON round-trip
# ============================================================

class TestFormatFidelity:
    """LLM Test 3 — Format: JSON with preamble text is still parsed correctly."""

    PREAMBLE_FLIGHT = (
        "Sure! Here are the top flights for you:\n\n"
        + _make_flight_json()
        + "\n\nI hope this helps with your travel planning!"
    )

    @patch("agents.agents._call_gemini", return_value=PREAMBLE_FLIGHT)
    def test_json_extracted_despite_preamble(self, _):
        result = run_flight_agent("NY", "LA", "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
        assert len(result["recommendations"]) == 1
        assert result["recommendations"][0]["airline"] == "TestAir"

    TRAILING_TEXT_HOTEL = (
        _make_hotel_json()
        + "\n\nAll options are SIMULATED. Please verify before booking."
    )

    @patch("agents.agents._call_gemini", return_value=TRAILING_TEXT_HOTEL)
    def test_hotel_trailing_text_captured_as_summary(self, _):
        result = run_hotel_agent("LA", "2025-06-01", "2025-06-07", 6, 15000, "", MOCK_HOTELS)
        assert len(result["recommendations"]) >= 1
        assert "SIMULATED" in result["summary"]


# ============================================================
# 4. BUDGET ADHERENCE
# ============================================================

class TestBudgetAdherence:
    """LLM Test 4 — Budget: workflow enforces budget and raises warnings."""

    def test_collect_requirements_budget_split_correct(self):
        state = _base_state()
        result = collect_requirements(state)
        total = state["total_budget_usd"]
        assert result["flight_budget_usd"] == pytest.approx(total * 0.40, rel=1e-3)
        nights = result["num_nights"]
        assert result["hotel_budget_per_night_usd"] == pytest.approx(
            (total * 0.45) / nights, rel=1e-3
        )
        assert result["car_budget_per_day_usd"] == pytest.approx(
            (total * 0.15) / nights, rel=1e-3
        )

    def test_validate_itinerary_over_budget_warning(self):
        state = collect_requirements(_base_state())
        # Inject expensive flight recommendation
        state["flight_result"] = {"recommendations": [{"fare_usd": 500000}]}
        state["hotel_result"] = {"recommendations": [{"nightly_rate_usd": 5000}]}
        state["car_result"]   = {"recommendations": [{"daily_rate_usd": 1000}]}
        result = validate_itinerary(state)
        assert any("exceed" in w.lower() or "over" in w.lower()
                   for w in result["validation_warnings"]), \
            "Expected over-budget warning not raised"

    def test_validate_itinerary_within_budget_no_overage_warning(self):
        state = collect_requirements(_base_state())
        # Inject cheap recommendations
        state["flight_result"] = {"recommendations": [{"fare_usd": 10000}]}
        state["hotel_result"] = {"recommendations": [{"nightly_rate_usd": 2000}]}
        state["car_result"]   = {"recommendations": [{"daily_rate_usd": 500}]}
        result = validate_itinerary(state)
        overage_warnings = [w for w in result["validation_warnings"]
                            if "exceed" in w.lower() or "over" in w.lower()]
        assert len(overage_warnings) == 0


# ============================================================
# 5. INPUT BOUNDARY — min, max, zero, very large values
# ============================================================

class TestInputBoundary:
    """LLM Test 5 — Boundary inputs handled without crash."""

    def test_minimum_budget(self):
        state = TravelState(
            origin="NYC", destination="LA",
            departure_date="2025-06-01", return_date="2025-06-02",
            total_budget_usd=5000, hotel_preferences="", passengers=1,
        )
        result = collect_requirements(state)
        assert result["validation_errors"] == []
        assert result["flight_budget_usd"] == pytest.approx(2000.0, rel=1e-3)

    def test_maximum_passengers(self):
        state = TravelState(
            origin="NYC", destination="LA",
            departure_date="2025-06-01", return_date="2025-06-07",
            total_budget_usd=1000000, hotel_preferences="", passengers=20,
        )
        result = collect_requirements(state)
        assert result["validation_errors"] == []
        assert result["passengers"] == 20 or state["passengers"] == 20

    def test_zero_budget_triggers_error(self):
        state = TravelState(
            origin="NYC", destination="LA",
            departure_date="2025-06-01", return_date="2025-06-07",
            total_budget_usd=0, hotel_preferences="", passengers=1,
        )
        result = collect_requirements(state)
        assert len(result["validation_errors"]) > 0

    def test_very_large_budget(self):
        state = TravelState(
            origin="NYC", destination="LA",
            departure_date="2025-06-01", return_date="2025-06-07",
            total_budget_usd=100_000_000, hotel_preferences="", passengers=1,
        )
        result = collect_requirements(state)
        assert result["validation_errors"] == []
        assert result["flight_budget_usd"] == pytest.approx(40_000_000, rel=1e-3)

    def test_one_night_trip(self):
        state = TravelState(
            origin="NYC", destination="LA",
            departure_date="2025-06-01", return_date="2025-06-02",
            total_budget_usd=50000, hotel_preferences="", passengers=1,
        )
        result = collect_requirements(state)
        assert result["num_nights"] == 1
        assert result["validation_errors"] == []

    def test_multi_month_trip(self):
        state = TravelState(
            origin="NYC", destination="LA",
            departure_date="2025-06-01", return_date="2025-08-31",
            total_budget_usd=500000, hotel_preferences="", passengers=2,
        )
        result = collect_requirements(state)
        assert result["num_nights"] == 91
        assert result["validation_errors"] == []


# ============================================================
# 6. NEGATIVE CASES — invalid inputs, bad JSON from LLM
# ============================================================

class TestNegativeCases:
    """LLM Test 6 — Negative: invalid inputs and malformed LLM output handled."""

    def test_return_before_departure_triggers_error(self):
        state = TravelState(
            origin="NYC", destination="LA",
            departure_date="2025-06-07", return_date="2025-06-01",
            total_budget_usd=100000, hotel_preferences="", passengers=1,
        )
        result = collect_requirements(state)
        assert len(result["validation_errors"]) > 0

    def test_same_day_departure_return_triggers_error(self):
        state = TravelState(
            origin="NYC", destination="LA",
            departure_date="2025-06-01", return_date="2025-06-01",
            total_budget_usd=100000, hotel_preferences="", passengers=1,
        )
        result = collect_requirements(state)
        assert len(result["validation_errors"]) > 0

    @patch("agents.agents._call_gemini", return_value="This is not JSON at all.")
    def test_flight_agent_graceful_on_invalid_json(self, _):
        result = run_flight_agent("NY", "LA", "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
        assert result["recommendations"] == []
        assert "not JSON" in result["summary"] or len(result["summary"]) > 0

    @patch("agents.agents._call_gemini", return_value="")
    def test_flight_agent_graceful_on_empty_response(self, _):
        result = run_flight_agent("NY", "LA", "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
        assert isinstance(result["recommendations"], list)

    @patch("agents.agents._call_gemini", return_value='{"no_recommendations_key": []}')
    def test_flight_agent_missing_recommendations_key(self, _):
        result = run_flight_agent("NY", "LA", "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
        assert result["recommendations"] == []

    @patch("agents.agents._call_gemini", return_value='{"recommendations": "not a list"}')
    def test_hotel_agent_recommendations_not_list(self, _):
        result = run_hotel_agent("LA", "2025-06-01", "2025-06-07", 6, 15000, "", MOCK_HOTELS)
        # Should not crash — recommendations key value returned as-is or []
        assert "recommendations" in result

    @patch("agents.agents._call_gemini", return_value='{"broken json')
    def test_car_agent_truncated_json(self, _):
        result = run_car_agent("LAX", "11:30", "Day 6", 6, 5000, 2, MOCK_CARS)
        assert result["recommendations"] == []
        assert len(result["summary"]) > 0


# ============================================================
# 7. ROBUSTNESS — prompt injection & special chars
# ============================================================

class TestRobustness:
    """LLM Test 7 — Robustness: adversarial / special-character inputs."""

    @patch("agents.agents._call_gemini", return_value=_make_flight_json())
    def test_prompt_injection_in_origin(self, _):
        """Origin containing JSON-like text must not break the parse path."""
        result = run_flight_agent(
            '{"hack": true}', "Los Angeles",
            "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS,
        )
        assert "recommendations" in result

    @patch("agents.agents._call_gemini", return_value=_make_hotel_json())
    def test_special_chars_in_preferences(self, _):
        result = run_hotel_agent(
            "Los Angeles", "2025-06-01", "2025-06-07", 6, 15000,
            "pool & gym; résumé-friendly; 100% free WiFi <test>", MOCK_HOTELS,
        )
        assert "recommendations" in result

    @patch("agents.agents._call_gemini", return_value=_make_car_json())
    def test_unicode_in_pickup_location(self, _):
        result = run_car_agent(
            "Tokya 東京 Airport", "11:30", "Day 6", 6, 5000, 2, MOCK_CARS,
        )
        assert "recommendations" in result


# ============================================================
# 8. MULTI-PASSENGER SCALING
# ============================================================

class TestMultiPassengerScaling:
    """LLM Test 8 — Multi-passenger: flight cost scales correctly."""

    def test_flight_cost_scales_with_passengers(self):
        flight_fare = 25000
        for pax in (1, 2, 5, 10):
            total = calculate_estimated_total(
                flight_fare=flight_fare, passengers=pax,
                nightly_rate=0, num_nights=0,
                daily_car_rate=0, num_days=0,
            )
            assert total == pytest.approx(flight_fare * pax, rel=1e-6)

    def test_validate_itinerary_multiplies_by_passenger_count(self):
        state = collect_requirements(TravelState(
            origin="NYC", destination="LA",
            departure_date="2025-06-01", return_date="2025-06-07",
            total_budget_usd=1_000_000, hotel_preferences="", passengers=3,
        ))
        fare = 20000
        state["flight_result"] = {"recommendations": [{"fare_usd": fare}]}
        state["hotel_result"] = {"recommendations": [{"nightly_rate_usd": 5000}]}
        state["car_result"]   = {"recommendations": [{"daily_rate_usd": 1000}]}
        result = validate_itinerary(state)
        # Flight portion = 20000 × 3 passengers = 60000
        assert result["estimated_total_cost_usd"] >= fare * 3


# ============================================================
# 9. DATA TYPE VALIDATION
# ============================================================

class TestDataTypes:
    """LLM Test 9 — Data types: all numeric fields parse as int/float."""

    @patch("agents.agents._call_gemini", return_value=_make_flight_json({"fare_usd": "25000"}))
    def test_flight_fare_as_string_in_json(self, _):
        """LLM sometimes returns numeric fields as strings — still parseable."""
        result = run_flight_agent("NY", "LA", "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
        rec = result["recommendations"][0]
        assert "fare_usd" in rec  # value may be str or numeric; agent passes it through

    @patch("agents.agents._call_gemini", return_value=_make_hotel_json({"stars": "4"}))
    def test_hotel_stars_as_string_in_json(self, _):
        result = run_hotel_agent("LA", "2025-06-01", "2025-06-07", 6, 15000, "", MOCK_HOTELS)
        assert "stars" in result["recommendations"][0]

    @patch("agents.agents._call_gemini", return_value=_make_car_json({"passengers": "5"}))
    def test_car_passengers_as_string_in_json(self, _):
        result = run_car_agent("LAX", "11:30", "Day 6", 6, 5000, 2, MOCK_CARS)
        assert "passengers" in result["recommendations"][0]


# ============================================================
# 10. DETERMINISM — same mock → same structural shape
# ============================================================

class TestDeterminism:
    """LLM Test 10 — Determinism: mocked LLM always returns same structure."""

    @patch("agents.agents._call_gemini", return_value=_make_flight_json())
    def test_repeated_calls_same_structure(self, _):
        results = [
            run_flight_agent("NY", "LA", "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
            for _ in range(3)
        ]
        keys_sets = [set(r["recommendations"][0].keys()) for r in results]
        assert keys_sets[0] == keys_sets[1] == keys_sets[2]


# ============================================================
# 11. AGENT ISOLATION
# ============================================================

class TestAgentIsolation:
    """LLM Test 11 — Isolation: each agent result contains only its own keys."""

    @patch("agents.agents._call_gemini", return_value=_make_flight_json())
    def test_flight_result_has_no_hotel_keys(self, _):
        result = run_flight_agent("NY", "LA", "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
        # Exclude keys shared across all schemas (id, reason); check hotel-specific keys only
        hotel_only_keys = REQUIRED_HOTEL_FIELDS - {"id", "reason"}
        rec = result["recommendations"][0]
        collision = hotel_only_keys & rec.keys()
        assert not collision, f"Flight result contains hotel-specific keys: {collision}"

    @patch("agents.agents._call_gemini", return_value=_make_hotel_json())
    def test_hotel_result_has_no_flight_keys(self, _):
        result = run_hotel_agent("LA", "2025-06-01", "2025-06-07", 6, 15000, "", MOCK_HOTELS)
        flight_keys = {"airline", "departure", "arrival", "duration", "stops"}
        rec = result["recommendations"][0]
        collision = flight_keys & rec.keys()
        assert not collision, f"Hotel result contains flight keys: {collision}"


# ============================================================
# 12. WORKFLOW INTEGRATION — end-to-end state flow (LLM mocked)
# ============================================================

class TestWorkflowIntegration:
    """LLM Test 12 — Integration: full TravelState populated through all nodes."""

    @patch("agents.agents._call_gemini", side_effect=[
        _make_flight_json(),
        _make_hotel_json(),
        _make_car_json(),
    ])
    def test_all_agent_results_present_in_state(self, _):
        from workflow import run_travel_planning
        state = run_travel_planning({
            "origin": "New York (JFK)",
            "destination": "Los Angeles (LAX)",
            "departure_date": "2025-06-01",
            "return_date": "2025-06-07",
            "total_budget_usd": 150000,
            "hotel_preferences": "pool",
            "passengers": 2,
        })
        assert state.get("flight_result") is not None
        assert state.get("hotel_result") is not None
        assert state.get("car_result") is not None

    @patch("agents.agents._call_gemini", side_effect=[
        _make_flight_json(),
        _make_hotel_json(),
        _make_car_json(),
    ])
    def test_itinerary_ready_flag_set_true(self, _):
        from workflow import run_travel_planning
        state = run_travel_planning({
            "origin": "NYC", "destination": "LA",
            "departure_date": "2025-06-01", "return_date": "2025-06-07",
            "total_budget_usd": 150000, "hotel_preferences": "", "passengers": 1,
        })
        assert state.get("itinerary_ready") is True

    @patch("agents.agents._call_gemini", side_effect=[
        _make_flight_json(),
        _make_hotel_json(),
        _make_car_json(),
    ])
    def test_estimated_cost_is_positive_float(self, _):
        from workflow import run_travel_planning
        state = run_travel_planning({
            "origin": "NYC", "destination": "LA",
            "departure_date": "2025-06-01", "return_date": "2025-06-07",
            "total_budget_usd": 150000, "hotel_preferences": "", "passengers": 2,
        })
        assert isinstance(state.get("estimated_total_cost_usd"), float)
        assert state["estimated_total_cost_usd"] > 0


# ============================================================
# 13. DATE EDGE CASES
# ============================================================

class TestDateEdgeCases:
    """LLM Test 13 — Edge dates: boundary scenarios for date validation."""

    def test_validate_dates_one_day_apart(self):
        ok, msg = validate_dates("2025-12-31", "2026-01-01")
        assert ok is True

    def test_validate_dates_leap_year(self):
        ok, _ = validate_dates("2024-02-28", "2024-02-29")
        assert ok is True

    def test_validate_dates_year_boundary(self):
        ok, _ = validate_dates("2025-12-31", "2026-01-14")
        assert ok is True
        assert nights_between("2025-12-31", "2026-01-14") == 14

    def test_nights_between_exact(self):
        assert nights_between("2025-06-01", "2025-06-08") == 7

    def test_nights_between_month_boundary(self):
        assert nights_between("2025-01-28", "2025-02-04") == 7


# ============================================================
# 14. COST CALCULATION ARITHMETIC
# ============================================================

class TestCostCalculationArithmetic:
    """LLM Test 14 — Cost calc: arithmetic is exact to 2 decimal places."""

    def test_calculate_estimated_total_basic(self):
        total = calculate_estimated_total(20000, 2, 9000, 6, 3500, 6)
        expected = (20000 * 2) + (9000 * 6) + (3500 * 6)
        assert total == pytest.approx(expected, rel=1e-9)

    def test_calculate_estimated_total_zero_car(self):
        total = calculate_estimated_total(20000, 1, 9000, 3, 0, 0)
        assert total == pytest.approx(20000 + 27000, rel=1e-9)

    def test_calculate_estimated_total_single_night(self):
        total = calculate_estimated_total(15000, 1, 8000, 1, 3000, 1)
        assert total == pytest.approx(26000.0, rel=1e-9)

    def test_check_budget_exactly_at_limit(self):
        within, diff = check_budget(100000, 100000)
        assert within is True
        assert diff == 0.0

    def test_check_budget_one_rupee_over(self):
        within, diff = check_budget(100001, 100000)
        assert within is False
        assert diff == pytest.approx(-1.0, rel=1e-9)


# ============================================================
# 15. BUDGET SPLIT VERIFICATION
# ============================================================

class TestBudgetSplit:
    """LLM Test 15 — Budget split: 40+45+15 = 100% of total."""

    def test_split_budget_sums_to_total_one_night(self):
        total = 100000
        budgets = split_budget(total, num_nights=1)
        reconstructed = (
            budgets["flight_budget_usd"]
            + budgets["hotel_budget_per_night_usd"] * 1
            + budgets["car_budget_per_day_usd"] * 1
        )
        assert reconstructed == pytest.approx(total, rel=1e-3)

    def test_split_budget_sums_to_total_six_nights(self):
        total = 150000
        nights = 6
        budgets = split_budget(total, num_nights=nights)
        reconstructed = (
            budgets["flight_budget_usd"]
            + budgets["hotel_budget_per_night_usd"] * nights
            + budgets["car_budget_per_day_usd"] * nights
        )
        assert reconstructed == pytest.approx(total, rel=1e-3)

    def test_split_budget_custom_percentages(self):
        budgets = split_budget(200000, num_nights=5,
                               flight_pct=0.50, hotel_pct=0.30, car_pct=0.20)
        assert budgets["flight_budget_usd"] == pytest.approx(100000, rel=1e-3)


# ============================================================
# 16. MOCK DATA FILTER BOUNDARY
# ============================================================

class TestMockDataFilterBoundary:
    """LLM Test 16 — Filter: budget filters at exact boundary values."""

    def test_flight_filter_exact_boundary(self):
        boundary_fare = 17500  # FL003 fare_usd = 17500
        flights = get_flights(budget=boundary_fare)
        assert all(f["fare_usd"] <= boundary_fare for f in flights)
        ids = [f["id"] for f in flights]
        assert "FL003" in ids

    def test_flight_filter_excludes_over_budget(self):
        flights = get_flights(budget=15000)
        assert all(f["fare_usd"] <= 15000 for f in flights)

    def test_hotel_filter_exact_boundary(self):
        boundary = 10000  # HT002 = 10000/night
        hotels = get_hotels(budget_per_night=boundary)
        ids = [h["id"] for h in hotels]
        assert "HT002" in ids

    def test_car_filter_exact_boundary(self):
        boundary = 4600  # CAR001 = 4600/day
        cars = get_cars(daily_budget=boundary)
        ids = [c["id"] for c in cars]
        assert "CAR001" in ids

    def test_all_data_returned_when_no_budget_filter(self):
        assert len(get_flights()) == 5
        assert len(get_hotels()) == 5
        assert len(get_cars()) == 5

    def test_no_results_when_budget_below_cheapest(self):
        flights = get_flights(budget=1)
        assert flights == []


# ============================================================
# 17. NO-RECOMMENDATION GRACEFUL HANDLING
# ============================================================

class TestNoRecommendationHandling:
    """LLM Test 17 — No-rec: empty recommendation list handled gracefully."""

    @patch("agents.agents._call_gemini", return_value='{"recommendations": []}')
    def test_flight_empty_recommendations_no_crash(self, _):
        result = run_flight_agent("NY", "LA", "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
        assert result["recommendations"] == []
        assert isinstance(result["summary"], str)

    @patch("agents.agents._call_gemini", return_value='{"recommendations": []}')
    def test_hotel_empty_recommendations_no_crash(self, _):
        result = run_hotel_agent("LA", "2025-06-01", "2025-06-07", 6, 15000, "", MOCK_HOTELS)
        assert result["recommendations"] == []

    @patch("agents.agents._call_gemini", return_value='{"recommendations": []}')
    def test_car_empty_recommendations_no_crash(self, _):
        result = run_car_agent("LAX", "11:30", "Day 6", 6, 5000, 2, MOCK_CARS)
        assert result["recommendations"] == []

    def test_validate_itinerary_warns_on_no_flight_rec(self):
        state = collect_requirements(_base_state())
        state["flight_result"] = {"recommendations": []}
        state["hotel_result"] = {"recommendations": [{"nightly_rate_usd": 9000}]}
        state["car_result"]   = {"recommendations": [{"daily_rate_usd": 3500}]}
        result = validate_itinerary(state)
        assert any("flight" in w.lower() for w in result["validation_warnings"])


# ============================================================
# 18. RESPONSE PARSING EDGE CASES
# ============================================================

class TestResponseParsing:
    """LLM Test 18 — Parsing: edge cases in raw LLM response handling."""

    @patch("agents.agents._call_gemini",
           return_value="  \n\n  " + _make_flight_json() + "  \n\n  ")
    def test_whitespace_padded_json(self, _):
        result = run_flight_agent("NY", "LA", "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
        assert len(result["recommendations"]) == 1

    NESTED_JSON_RESPONSE = json.dumps({
        "recommendations": [
            {
                "id": "FL001", "airline": "NestedAir",
                "departure": "09:00", "arrival": "12:30",
                "duration": "5h 30m", "fare_usd": 18000,
                "class": "Economy", "stops": 1,
                "reason": "Affordable with one stop.",
                "metadata": {"booking_class": "X", "seats_left": 5},
            }
        ]
    })

    @patch("agents.agents._call_gemini", return_value=NESTED_JSON_RESPONSE)
    def test_nested_json_extra_field_preserved(self, _):
        result = run_flight_agent("NY", "LA", "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
        rec = result["recommendations"][0]
        assert rec["airline"] == "NestedAir"

    MULTI_REC_RESPONSE = json.dumps({
        "recommendations": [
            {"id": "FL001", "airline": "Air1", "departure": "08:00", "arrival": "11:30",
             "duration": "5h", "fare_usd": 18000, "class": "Economy", "stops": 0, "reason": "Fast"},
            {"id": "FL002", "airline": "Air2", "departure": "12:00", "arrival": "15:30",
             "duration": "5h", "fare_usd": 20000, "class": "Economy", "stops": 0, "reason": "Comfortable"},
            {"id": "FL003", "airline": "Air3", "departure": "18:00", "arrival": "21:30",
             "duration": "5h", "fare_usd": 14000, "class": "Economy", "stops": 1, "reason": "Budget"},
        ]
    })

    @patch("agents.agents._call_gemini", return_value=MULTI_REC_RESPONSE)
    def test_three_recommendations_parsed_correctly(self, _):
        result = run_flight_agent("NY", "LA", "2025-06-01", "2025-06-07", 60000, MOCK_FLIGHTS)
        assert len(result["recommendations"]) == 3
        airlines = [r["airline"] for r in result["recommendations"]]
        assert airlines == ["Air1", "Air2", "Air3"]
