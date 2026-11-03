"""
Mock datasets for flights, hotels, and rental cars.
These simulate provider data so the system works without paid travel APIs.
All options are clearly labelled as SIMULATED.
"""

from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Mock Flights
# ---------------------------------------------------------------------------
MOCK_FLIGHTS: List[Dict[str, Any]] = [
    {
        "id": "FL001",
        "airline": "SkyWings Airlines",
        "origin": "New York (JFK)",
        "destination": "Los Angeles (LAX)",
        "departure": "08:00",
        "arrival": "11:30",
        "duration": "5h 30m",
        "fare_usd": 27000,
        "class": "Economy",
        "stops": 0,
        "simulated": True,
    },
    {
        "id": "FL002",
        "airline": "BlueHorizon Air",
        "origin": "New York (JFK)",
        "destination": "Los Angeles (LAX)",
        "departure": "11:45",
        "arrival": "15:20",
        "duration": "5h 35m",
        "fare_usd": 23500,
        "class": "Economy",
        "stops": 0,
        "simulated": True,
    },
    {
        "id": "FL003",
        "airline": "SkyWings Airlines",
        "origin": "New York (JFK)",
        "destination": "Los Angeles (LAX)",
        "departure": "18:00",
        "arrival": "21:45",
        "duration": "5h 45m",
        "fare_usd": 17500,
        "class": "Economy",
        "stops": 1,
        "simulated": True,
    },
    {
        "id": "FL004",
        "airline": "PremiumJet",
        "origin": "New York (JFK)",
        "destination": "Los Angeles (LAX)",
        "departure": "07:00",
        "arrival": "10:25",
        "duration": "5h 25m",
        "fare_usd": 57000,
        "class": "Business",
        "stops": 0,
        "simulated": True,
    },
    {
        "id": "FL005",
        "airline": "BudgetFly",
        "origin": "New York (JFK)",
        "destination": "Los Angeles (LAX)",
        "departure": "23:55",
        "arrival": "03:40+1",
        "duration": "5h 45m",
        "fare_usd": 13500,
        "class": "Economy",
        "stops": 1,
        "simulated": True,
    },
]

# ---------------------------------------------------------------------------
# Mock Hotels
# ---------------------------------------------------------------------------
MOCK_HOTELS: List[Dict[str, Any]] = [
    {
        "id": "HT001",
        "name": "Grand Skyline Hotel",
        "destination": "Los Angeles",
        "stars": 5,
        "nightly_rate_usd": 27000,
        "location": "Downtown LA",
        "amenities": ["Pool", "Gym", "Spa", "Free WiFi", "Breakfast included"],
        "rating": 4.8,
        "simulated": True,
    },
    {
        "id": "HT002",
        "name": "Comfort Inn Central",
        "destination": "Los Angeles",
        "stars": 3,
        "nightly_rate_usd": 10000,
        "location": "Mid-Wilshire",
        "amenities": ["Free WiFi", "Parking", "24h Front Desk"],
        "rating": 4.1,
        "simulated": True,
    },
    {
        "id": "HT003",
        "name": "Beachside Retreat",
        "destination": "Los Angeles",
        "stars": 4,
        "nightly_rate_usd": 20000,
        "location": "Santa Monica",
        "amenities": ["Beach Access", "Pool", "Restaurant", "Free WiFi"],
        "rating": 4.6,
        "simulated": True,
    },
    {
        "id": "HT004",
        "name": "Budget Stay Express",
        "destination": "Los Angeles",
        "stars": 2,
        "nightly_rate_usd": 5900,
        "location": "North Hollywood",
        "amenities": ["Free WiFi", "Parking"],
        "rating": 3.5,
        "simulated": True,
    },
    {
        "id": "HT005",
        "name": "Urban Boutique Hotel",
        "destination": "Los Angeles",
        "stars": 4,
        "nightly_rate_usd": 16500,
        "location": "Hollywood",
        "amenities": ["Rooftop Bar", "Gym", "Free WiFi", "Concierge"],
        "rating": 4.4,
        "simulated": True,
    },
]

# ---------------------------------------------------------------------------
# Mock Rental Cars
# ---------------------------------------------------------------------------
MOCK_CARS: List[Dict[str, Any]] = [
    {
        "id": "CAR001",
        "provider": "AutoRent Premier",
        "vehicle": "Toyota Camry",
        "category": "Sedan",
        "daily_rate_usd": 4600,
        "pickup_location": "LAX Airport",
        "fuel_type": "Petrol",
        "passengers": 5,
        "simulated": True,
    },
    {
        "id": "CAR002",
        "provider": "EcoDrive",
        "vehicle": "Tesla Model 3",
        "category": "Electric Sedan",
        "daily_rate_usd": 8000,
        "pickup_location": "LAX Airport",
        "fuel_type": "Electric",
        "passengers": 5,
        "simulated": True,
    },
    {
        "id": "CAR003",
        "provider": "ValueWheels",
        "vehicle": "Hyundai Elantra",
        "category": "Economy",
        "daily_rate_usd": 3200,
        "pickup_location": "LAX Airport",
        "fuel_type": "Petrol",
        "passengers": 5,
        "simulated": True,
    },
    {
        "id": "CAR004",
        "provider": "LuxuryCruise",
        "vehicle": "BMW 5 Series",
        "category": "Luxury",
        "daily_rate_usd": 14700,
        "pickup_location": "LAX Airport",
        "fuel_type": "Petrol",
        "passengers": 5,
        "simulated": True,
    },
    {
        "id": "CAR005",
        "provider": "FamilyDrive",
        "vehicle": "Chrysler Pacifica",
        "category": "Minivan",
        "daily_rate_usd": 7100,
        "pickup_location": "LAX Airport",
        "fuel_type": "Petrol/Hybrid",
        "passengers": 7,
        "simulated": True,
    },
]


def get_flights(origin: str = "", destination: str = "", budget: float = 0) -> List[Dict]:
    """Return mock flights filtered by budget (0 = no filter)."""
    flights = MOCK_FLIGHTS
    if budget > 0:
        flights = [f for f in flights if f["fare_usd"] <= budget]
    return flights


def get_hotels(destination: str = "", budget_per_night: float = 0) -> List[Dict]:
    """Return mock hotels filtered by nightly rate budget (0 = no filter)."""
    hotels = MOCK_HOTELS
    if budget_per_night > 0:
        hotels = [h for h in hotels if h["nightly_rate_usd"] <= budget_per_night]
    return hotels


def get_cars(daily_budget: float = 0) -> List[Dict]:
    """Return mock cars filtered by daily budget (0 = no filter)."""
    cars = MOCK_CARS
    if daily_budget > 0:
        cars = [c for c in cars if c["daily_rate_usd"] <= daily_budget]
    return cars
