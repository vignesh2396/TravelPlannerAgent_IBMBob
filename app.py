"""
Streamlit UI for the Intelligent Travel Planning Multi-Agent System.

Run with:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

# Add the project root to sys.path so local modules resolve correctly
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from workflow import run_travel_planning

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Travel Planner — Multi-Agent System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — minimal, clean
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .agent-card {
        background: #f0f4ff;
        border-left: 4px solid #3b82d4;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .error-box {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .success-box {
        background: #dcfce7;
        border-left: 4px solid #22c55e;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .disclaimer {
        font-size: 0.8em;
        color: #57606a;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar — travel requirements form
# ---------------------------------------------------------------------------
def render_sidebar() -> dict | None:
    st.sidebar.title("✈️ Travel Requirements")
    st.sidebar.markdown(
        "<span class='disclaimer'>⚠️ All results are SIMULATED mock data "
        "for demonstration purposes only.</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    with st.sidebar.form("travel_form"):
        origin = st.text_input("Origin city / airport", value="New York (JFK)")
        destination = st.text_input("Destination city / airport", value="Los Angeles (LAX)")

        col1, col2 = st.columns(2)
        with col1:
            departure = st.date_input("Departure date")
        with col2:
            return_d = st.date_input("Return date")

        total_budget = st.number_input(
            "Total trip budget (INR)", min_value=5000, value=150000, step=5000
        )
        passengers = st.number_input("Passengers", min_value=1, max_value=20, value=1)
        preferences = st.text_input(
            "Hotel preferences (optional)", placeholder="e.g. pool, free WiFi, central"
        )

        submitted = st.form_submit_button("🚀 Plan My Trip", use_container_width=True)

    if submitted:
        if return_d <= departure:
            st.sidebar.error("Return date must be after departure date.")
            return None
        return {
            "origin": origin,
            "destination": destination,
            "departure_date": str(departure),
            "return_date": str(return_d),
            "total_budget_usd": total_budget,
            "hotel_preferences": preferences,
            "passengers": passengers,
        }
    return None


# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------


def render_flight_results(result: dict) -> None:
    st.subheader("✈️ Agent 1 — Flight Booking Agent")
    st.markdown(
        "<div class='disclaimer'>Results powered by Google Gemini · SIMULATED options</div>",
        unsafe_allow_html=True,
    )

    recs = result.get("recommendations", [])
    if recs:
        for i, flight in enumerate(recs, 1):
            with st.expander(
                f"#{i} {flight.get('airline', 'N/A')} — "
                f"₹{flight.get('fare_usd', 'N/A')} · {flight.get('class', '')} · "
                f"{flight.get('stops', 0)} stop(s)",
                expanded=(i == 1),
            ):
                cols = st.columns(3)
                cols[0].metric("Departure", flight.get("departure", "—"))
                cols[1].metric("Arrival", flight.get("arrival", "—"))
                cols[2].metric("Duration", flight.get("duration", "—"))
                st.info(f"💡 {flight.get('reason', '')}")
    else:
        st.markdown(
            "<div class='warning-box'>No structured flight recommendations extracted. "
            "See summary below.</div>",
            unsafe_allow_html=True,
        )

    if result.get("summary"):
        with st.expander("Agent summary", expanded=not bool(recs)):
            st.write(result["summary"])


def render_hotel_results(result: dict) -> None:
    st.subheader("🏨 Agent 2 — Hotel Booking Agent")
    st.markdown(
        "<div class='disclaimer'>Results powered by Google Gemini · SIMULATED options</div>",
        unsafe_allow_html=True,
    )

    recs = result.get("recommendations", [])
    if recs:
        for i, hotel in enumerate(recs, 1):
            with st.expander(
                f"#{i} {hotel.get('name', 'N/A')} — "
                f"⭐ {hotel.get('stars', '?')} · "
                f"₹{hotel.get('nightly_rate_usd', 'N/A')}/night · "
                f"Total ₹{hotel.get('total_cost_usd', 'N/A')}",
                expanded=(i == 1),
            ):
                cols = st.columns(3)
                cols[0].metric("Location", hotel.get("location", "—"))
                cols[1].metric("Rating", f"{hotel.get('rating', '—')} / 5")
                cols[2].metric("Nightly Rate", f"₹{hotel.get('nightly_rate_usd', '—')}")
                amenities = hotel.get("amenities", [])
                if amenities:
                    st.write("**Amenities:** " + ", ".join(amenities))
                st.info(f"💡 {hotel.get('reason', '')}")
    else:
        st.markdown(
            "<div class='warning-box'>No structured hotel recommendations extracted. "
            "See summary below.</div>",
            unsafe_allow_html=True,
        )

    if result.get("summary"):
        with st.expander("Agent summary", expanded=not bool(recs)):
            st.write(result["summary"])


def render_car_results(result: dict) -> None:
    st.subheader("🚗 Agent 3 — Car Booking Agent")
    st.markdown(
        "<div class='disclaimer'>Results powered by Google Gemini · SIMULATED options</div>",
        unsafe_allow_html=True,
    )

    recs = result.get("recommendations", [])
    if recs:
        for i, car in enumerate(recs, 1):
            with st.expander(
                f"#{i} {car.get('vehicle', 'N/A')} ({car.get('category', '')}) — "
                f"₹{car.get('daily_rate_usd', 'N/A')}/day · "
                f"Total ₹{car.get('total_cost_usd', 'N/A')}",
                expanded=(i == 1),
            ):
                cols = st.columns(3)
                cols[0].metric("Provider", car.get("provider", "—"))
                cols[1].metric("Passengers", car.get("passengers", "—"))
                cols[2].metric("Pickup", car.get("pickup_location", "—"))
                st.info(f"💡 {car.get('reason', '')}")
    else:
        st.markdown(
            "<div class='warning-box'>No structured car recommendations extracted. "
            "See summary below.</div>",
            unsafe_allow_html=True,
        )

    if result.get("summary"):
        with st.expander("Agent summary", expanded=not bool(recs)):
            st.write(result["summary"])


def render_cost_summary(state: dict) -> None:
    total = state.get("estimated_total_cost_usd", 0)
    budget = state.get("total_budget_usd", 0)
    warnings = state.get("validation_warnings", [])

    st.subheader("💰 Cost Summary")
    cols = st.columns(3)
    cols[0].metric("Estimated Total", f"₹{total:,.0f}")
    cols[1].metric("Your Budget", f"₹{budget:,.0f}")
    remaining = budget - total
    cols[2].metric(
        "Remaining / Overage",
        f"₹{abs(remaining):,.0f}",
        delta=f"{'under' if remaining >= 0 else 'over'} budget",
        delta_color="normal" if remaining >= 0 else "inverse",
    )

    for w in warnings:
        st.markdown(
            f"<div class='warning-box'>⚠️ {w}</div>", unsafe_allow_html=True
        )


def render_approval_section(state: dict) -> bool:
    st.markdown("---")
    st.subheader("✅ Review & Approve")
    st.markdown(
        "<div class='disclaimer'>"
        "⚠️ This is a SIMULATED booking system. "
        "Approving will generate a simulated confirmation only — "
        "no real payment or reservation will be made."
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    approved = col1.button("✅ Approve & Simulate Booking", type="primary", use_container_width=True)
    cancelled = col2.button("❌ Cancel", use_container_width=True)

    if approved:
        return True
    if cancelled:
        st.session_state.clear()
        st.rerun()
    return False


def render_confirmation(state: dict) -> None:
    st.balloons()
    st.markdown(
        "<div class='success-box'>"
        "✅ <strong>Simulated booking confirmed!</strong><br>"
        "In a production system this would trigger real provider API calls. "
        "No real payment or booking has been processed."
        "</div>",
        unsafe_allow_html=True,
    )
    st.metric("Simulated Total Cost", f"₹{state.get('estimated_total_cost_usd', 0):,.0f}")
    st.info("Confirmation references: FL-SIM-AUTO | HT-SIM-AUTO | CAR-SIM-AUTO")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("✈️ AI Travel Planning — Multi-Agent System")
    st.markdown(
        "Three specialized AI agents (**Flight · Hotel · Car**) powered by "
        "**Google Gemini** collaborate to build your coordinated travel itinerary.\n\n"
        "> ⚠️ **All data shown is SIMULATED (mock).** "
        "This is a prototype — no real bookings are made."
    )

    # Check API key early
    if not os.environ.get("GEMINI_API_KEY"):
        st.error(
            "🔑 **GEMINI_API_KEY not found.** "
            "Create a `.env` file in the project root with:\n\n"
            "```\nGEMINI_API_KEY=your_key_here\n```\n\n"
            "Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey)."
        )
        st.stop()

    inputs = render_sidebar()

    if inputs:
        st.session_state["inputs"] = inputs
        st.session_state["state"] = None
        st.session_state["confirmed"] = False

    if "confirmed" in st.session_state and st.session_state["confirmed"]:
        render_confirmation(st.session_state.get("state", {}))
        if st.button("🔄 Plan Another Trip"):
            st.session_state.clear()
            st.rerun()
        return

    if "inputs" not in st.session_state:
        st.info("👈 Fill in your travel requirements in the sidebar and click **Plan My Trip**.")
        return

    # Run workflow (or use cached result)
    if st.session_state.get("state") is None:
        inp = st.session_state["inputs"]
        with st.spinner("🤖 Three AI agents are building your itinerary…"):
            try:
                state = run_travel_planning(inp)
                st.session_state["state"] = state
            except EnvironmentError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"An error occurred: {exc}")
                return

    state = st.session_state["state"]

    # Validation errors
    for err in state.get("validation_errors", []):
        st.markdown(f"<div class='error-box'>❌ {err}</div>", unsafe_allow_html=True)

    if state.get("error") and not state.get("itinerary_ready"):
        return

    # Agent results
    tab1, tab2, tab3 = st.tabs(["✈️ Flights", "🏨 Hotels", "🚗 Cars"])

    with tab1:
        if state.get("flight_result"):
            render_flight_results(state["flight_result"])

    with tab2:
        if state.get("hotel_result"):
            render_hotel_results(state["hotel_result"])

    with tab3:
        if state.get("car_result"):
            render_car_results(state["car_result"])

    st.markdown("---")
    render_cost_summary(state)

    if state.get("itinerary_ready"):
        if render_approval_section(state):
            st.session_state["confirmed"] = True
            st.rerun()


if __name__ == "__main__":
    main()
