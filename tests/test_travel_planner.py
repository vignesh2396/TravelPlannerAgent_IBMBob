"""
Unit tests for the Travel Planning Multi-Agent System.

Tests cover:
  - Mock data filters
  - Utility functions (date, budget)
  - Workflow validation (collect_requirements node)
  - Agent stubs (run agents with a monkey-patched LLM)

Run with:
    pytest tests/ -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.mock_data import get_cars, get_flights, get_hotels
from utils import (
    calculate_estimated_total,
    check_budget,
    nights_between,
    split_budget,
    validate_dates,
)
from workflow import collect_requirements, TravelState


# ---------------------------------------------------------------------------
# Mock data tests
# ---------------------------------------------------------------------------


class TestMockData:
    def test_get_flights_no_filter(self):
        flights = get_flights()
        assert len(flights) == 5
        assert all(f["simulated"] for f in flights)

    def test_get_flights_budget_filter(self):
        flights = get_flights(budget=250)
        assert all(f["fare_usd"] <= 250 for f in flights)
        assert len(flights) > 0

    def test_get_hotels_no_filter(self):
        hotels = get_hotels()
        assert len(hotels) == 5

    def test_get_hotels_budget_filter(self):
        hotels = get_hotels(budget_per_night=150)
        assert all(h["nightly_rate_usd"] <= 150 for h in hotels)

    def test_get_cars_no_filter(self):
        cars = get_cars()
        assert len(cars) == 5

    def test_get_cars_budget_filter(self):
        cars = get_cars(daily_budget=60)
        assert all(c["daily_rate_usd"] <= 60 for c in cars)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestUtils:
    def test_nights_between(self):
        assert nights_between("2025-06-01", "2025-06-07") == 6

    def test_validate_dates_valid(self):
        ok, msg = validate_dates("2025-06-01", "2025-06-07")
        assert ok is True
        assert msg == ""

    def test_validate_dates_same_day(self):
        ok, msg = validate_dates("2025-06-01", "2025-06-01")
        assert ok is False

    def test_validate_dates_return_before_departure(self):
        ok, msg = validate_dates("2025-06-07", "2025-06-01")
        assert ok is False

    def test_validate_dates_bad_format(self):
        ok, msg = validate_dates("01/06/2025", "07/06/2025")
        assert ok is False

    def test_split_budget(self):
        budgets = split_budget(total=2000, num_nights=6)
        assert budgets["flight_budget_usd"] == pytest.approx(800, rel=1e-3)
        assert abs(budgets["hotel_budget_per_night_usd"] - 150) < 1
        assert abs(budgets["car_budget_per_day_usd"] - 50) < 1

    def test_calculate_estimated_total(self):
        total = calculate_estimated_total(
            flight_fare=300, passengers=2,
            nightly_rate=150, num_nights=5,
            daily_car_rate=50, num_days=5,
        )
        assert total == 600 + 750 + 250  # 1600

    def test_check_budget_within(self):
        within, diff = check_budget(1500, 2000)
        assert within is True
        assert diff == 500

    def test_check_budget_over(self):
        within, diff = check_budget(2500, 2000)
        assert within is False
        assert diff == -500


# ---------------------------------------------------------------------------
# Workflow node tests
# ---------------------------------------------------------------------------


class TestCollectRequirements:
    def _base_state(self) -> TravelState:
        return TravelState(
            origin="New York",
            destination="Los Angeles",
            departure_date="2025-06-01",
            return_date="2025-06-07",
            total_budget_usd=2000,
            hotel_preferences="",
            passengers=2,
        )

    def test_happy_path(self):
        result = collect_requirements(self._base_state())
        assert result["validation_errors"] == []
        assert result["num_nights"] == 6
        assert result["flight_budget_usd"] == pytest.approx(800, rel=1e-3)
        assert result["error"] == ""

    def test_invalid_dates(self):
        state = self._base_state()
        state["return_date"] = "2025-05-01"
        result = collect_requirements(state)
        assert len(result["validation_errors"]) > 0

    def test_zero_budget(self):
        state = self._base_state()
        state["total_budget_usd"] = 0
        result = collect_requirements(state)
        assert any("budget" in e.lower() for e in result["validation_errors"])


# ---------------------------------------------------------------------------
# Agent smoke-tests (LLM call is mocked)
# ---------------------------------------------------------------------------

MOCK_LLM_RESPONSE = json.dumps(
    {
        "recommendations": [
            {
                "id": "FL001",
                "airline": "MockAir",
                "departure": "08:00",
                "arrival": "11:30",
                "duration": "5h 30m",
                "fare_usd": 280,
                "class": "Economy",
                "stops": 0,
                "reason": "Best value for budget.",
            }
        ]
    }
)


class TestFlightAgent:
    @patch("agents.agents._call_gemini", return_value=MOCK_LLM_RESPONSE)
    def test_returns_recommendations(self, _mock):
        from agents.agents import run_flight_agent
        from data.mock_data import MOCK_FLIGHTS

        result = run_flight_agent(
            origin="New York",
            destination="Los Angeles",
            departure_date="2025-06-01",
            return_date="2025-06-07",
            budget_usd=500,
            flights=MOCK_FLIGHTS,
        )
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)


MOCK_HOTEL_RESPONSE = json.dumps(
    {
        "recommendations": [
            {
                "id": "HT002",
                "name": "Mock Hotel",
                "stars": 3,
                "nightly_rate_usd": 120,
                "total_cost_usd": 720,
                "location": "Downtown",
                "rating": 4.1,
                "amenities": ["WiFi"],
                "reason": "Great value.",
            }
        ]
    }
)


class TestHotelAgent:
    @patch("agents.agents._call_gemini", return_value=MOCK_HOTEL_RESPONSE)
    def test_returns_recommendations(self, _mock):
        from agents.agents import run_hotel_agent
        from data.mock_data import MOCK_HOTELS

        result = run_hotel_agent(
            destination="Los Angeles",
            check_in="2025-06-01",
            check_out="2025-06-07",
            num_nights=6,
            budget_per_night_usd=150,
            preferences="",
            hotels=MOCK_HOTELS,
        )
        assert "recommendations" in result
        assert len(result["recommendations"]) >= 1


MOCK_CAR_RESPONSE = json.dumps(
    {
        "recommendations": [
            {
                "id": "CAR001",
                "provider": "MockRent",
                "vehicle": "Toyota Camry",
                "category": "Sedan",
                "daily_rate_usd": 55,
                "total_cost_usd": 330,
                "pickup_location": "LAX",
                "passengers": 5,
                "reason": "Reliable and affordable.",
            }
        ]
    }
)


class TestCarAgent:
    @patch("agents.agents._call_gemini", return_value=MOCK_CAR_RESPONSE)
    def test_returns_recommendations(self, _mock):
        from agents.agents import run_car_agent
        from data.mock_data import MOCK_CARS

        result = run_car_agent(
            pickup_location="LAX",
            pickup_time="11:30",
            dropoff_time="Day 6 evening",
            num_days=6,
            budget_per_day_usd=60,
            passengers=2,
            cars=MOCK_CARS,
        )
        assert "recommendations" in result
