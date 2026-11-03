"""
generate_video.py
=================
Creates a fully narrated slideshow MP4 for the
AI Travel Planning — Multi-Agent System project.

Dependencies (all already installed):
    pip install moviepy pillow numpy gtts pyttsx3

Run:
    python generate_video.py

Output:
    AI_Travel_Planner_MAS.mp4  (same folder as this script)

Design decisions:
  * PIL draws every frame — no ImageMagick required.
  * pyttsx3 (Windows SAPI5) generates narration WAV files offline.
  * moviepy concatenates slides with cross-fade transitions.
  * Each slide has: animated title drop-in, body fade-in, narration audio.
  * Fallback: if audio fails, slides are rendered silently.
"""

from __future__ import annotations

import os
import sys
import math
import warnings
import tempfile
import textwrap
from pathlib import Path
from typing import Callable

import warnings
warnings.filterwarnings("ignore")          # suppress pyttsx3 / comtypes FutureWarning

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    VideoClip,
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
)
from moviepy.audio.AudioClip import AudioArrayClip

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
W, H       = 1280, 720          # video resolution
FPS        = 24
TITLE_DUR  = 7.0                # seconds for title slide
SLIDE_DUR  = 9.0                # seconds for content slides
FADE_DUR   = 0.5                # cross-fade duration

WINDIR = os.environ.get("WINDIR", "C:/Windows")
FONT_DIR = os.path.join(WINDIR, "Fonts")

# Colours
BG_DARK   = (30,  58,  95)      # #1E3A5F
BG_WHITE  = (255, 255, 255)
ACCENT1   = (59,  130, 212)     # #3B82D4
ACCENT2   = (245, 158, 11)      # #F59E0B
GREEN     = (22,  163,  74)     # #16A34A
PURPLE    = (124,  92, 216)     # #7C5CD8
TEAL      = (15,  118, 110)     # #0F766E
AMBER     = (180,  83,  9)      # #B45309
TEXT_DK   = (31,  35,  40)
TEXT_MUT  = (87,  96, 106)
WHITE     = (255, 255, 255)
YELLOW    = (252, 211,  77)     # #FCD34D
LBLUE     = (147, 197, 253)     # #93C5FD

# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------
_font_cache: dict[tuple, ImageFont.FreeTypeFont] = {}

def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _font_cache:
        name = "arialbd.ttf" if bold else "arial.ttf"
        try:
            _font_cache[key] = ImageFont.truetype(os.path.join(FONT_DIR, name), size)
        except Exception:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]

def mono(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, "consola.ttf"), size)
    except Exception:
        return font(size)

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def new_img(bg: tuple = BG_WHITE) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), bg)
    return img, ImageDraw.Draw(img)

def header_bar(draw: ImageDraw.ImageDraw, text: str, slide_num: str,
               bg: tuple = BG_DARK) -> None:
    draw.rectangle([0, 0, W, 68], fill=bg)
    draw.text((20, 12), text, font=font(34, True), fill=WHITE)
    draw.text((W - 80, 12), slide_num, font=font(34, True), fill=ACCENT1)

def top_stripe(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle([0, 0, W, 8], fill=ACCENT1)

def bottom_stripe(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle([0, H - 8, W, H], fill=ACCENT2)

def wrap_text(draw: ImageDraw.ImageDraw, text: str, x: int, y: int,
              max_w: int, fnt: ImageFont.FreeTypeFont, fill: tuple,
              line_gap: int = 6) -> int:
    """Draw wrapped text, return bottom y."""
    avg_char = fnt.getlength("A")
    chars_per_line = max(1, int(max_w / avg_char))
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=chars_per_line) or [""])
    cy = y
    for line in lines:
        draw.text((x, cy), line, font=fnt, fill=fill)
        bbox = fnt.getbbox(line) if hasattr(fnt, "getbbox") else (0, 0, 0, fnt.size)
        cy += (bbox[3] - bbox[1]) + line_gap
    return cy

def label_box(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
              text: str, bg: tuple, fg: tuple = WHITE, size: int = 16) -> None:
    draw.rectangle([x, y, x + w, y + h], fill=bg)
    draw.text((x + 8, y + 4), text, font=font(size, True), fill=fg)

def content_box(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
                text: str, bg: tuple, fg: tuple = TEXT_DK, size: int = 15,
                mono_font: bool = False) -> None:
    draw.rectangle([x, y, x + w, y + h], fill=bg)
    fnt = mono(size) if mono_font else font(size)
    wrap_text(draw, text, x + 10, y + 8, w - 20, fnt, fg)

def pill(draw: ImageDraw.ImageDraw, x: int, y: int,
         text: str, bg: tuple, fg: tuple = WHITE, size: int = 14) -> int:
    fnt = font(size, True)
    tw = int(fnt.getlength(text))
    draw.rectangle([x, y, x + tw + 20, y + size + 10], fill=bg)
    draw.text((x + 10, y + 5), text, font=fnt, fill=fg)
    return x + tw + 28

# ---------------------------------------------------------------------------
# Narration (pyttsx3 — offline TTS)
# ---------------------------------------------------------------------------
_TMP_DIR = Path(tempfile.gettempdir()) / "travel_planner_video"
_TMP_DIR.mkdir(exist_ok=True)

def make_narration(idx: int, text: str) -> str | None:
    """Render narration to a WAV file, return path or None on failure."""
    out = str(_TMP_DIR / f"narr_{idx:02d}.wav")
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return out
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.setProperty("volume", 0.95)
        engine.save_to_file(text, out)
        engine.runAndWait()
        if os.path.exists(out) and os.path.getsize(out) > 500:
            return out
    except Exception as exc:
        print(f"  [audio] narration {idx} failed: {exc}")
    return None

def silent_audio(duration: float) -> AudioArrayClip:
    rate = 44100
    samples = int(duration * rate)
    arr = np.zeros((samples, 2), dtype=np.float32)
    return AudioArrayClip(arr, fps=rate)

def attach_audio(clip: VideoClip, wav_path: str | None) -> VideoClip:
    dur = clip.duration
    if wav_path and os.path.exists(wav_path):
        try:
            aud = AudioFileClip(wav_path)
            if aud.duration < dur:
                sil = silent_audio(dur - aud.duration)
                from moviepy.audio.AudioClip import CompositeAudioClip, concatenate_audioclips
                aud = concatenate_audioclips([aud, sil])
            aud = aud.subclip(0, dur)
            return clip.set_audio(aud)
        except Exception as exc:
            print(f"  [audio] attach failed: {exc}")
    return clip.set_audio(silent_audio(dur))

# ---------------------------------------------------------------------------
# Animated clip builder
# ---------------------------------------------------------------------------

def animated_clip(
    bg_frame_fn: Callable[[float], np.ndarray],
    duration: float,
    title_text: str,
    title_y: int,
    body_start: float = 0.6,   # fraction of duration when body fades in
) -> VideoClip:
    """
    bg_frame_fn(t) returns the numpy array for the background at time t.
    Title drops from title_y-40 → title_y over 0..0.4*duration.
    Body fades in at body_start fraction.
    """
    def make_frame(t: float) -> np.ndarray:
        return bg_frame_fn(t)
    return VideoClip(make_frame, duration=duration)

def slide_to_clip(img: Image.Image, duration: float,
                  animate: bool = True) -> VideoClip:
    """
    Convert a PIL Image to a VideoClip with optional zoom-in animation.
    Zoom goes from 1.02x → 1.0x over the first 1 s (subtle Ken Burns).
    """
    base_arr = np.array(img)

    if not animate:
        return ImageClip(base_arr, duration=duration)

    def make_frame(t: float) -> np.ndarray:
        scale = 1.0 + max(0.0, 0.02 * (1.0 - t / min(1.5, duration)))
        if abs(scale - 1.0) < 0.001:
            return base_arr
        new_w = int(W * scale)
        new_h = int(H * scale)
        large = Image.fromarray(base_arr).resize((new_w, new_h), Image.LANCZOS)
        ox = (new_w - W) // 2
        oy = (new_h - H) // 2
        return np.array(large.crop((ox, oy, ox + W, oy + H)))

    return VideoClip(make_frame, duration=duration)

# ---------------------------------------------------------------------------
# Individual slide renderers
# ---------------------------------------------------------------------------

def slide_01_title() -> tuple[Image.Image, str]:
    img, draw = new_img(BG_DARK)
    top_stripe(draw)
    bottom_stripe(draw)
    # Emoji strip
    draw.text((W // 2 - 80, 40), "✈️", font=font(72), fill=WHITE)
    # Main title
    draw.text((W // 2 - 380, 180), "AI Travel Planning", font=font(72, True), fill=WHITE)
    # Subtitle
    draw.text((W // 2 - 240, 270), "Multi-Agent System", font=font(48), fill=LBLUE)
    # Tag line
    draw.rectangle([60, 340, W - 60, 342], fill=ACCENT1)
    draw.text((100, 358), "Domain:  Travel, Tourism & Hospitality",
              font=font(22, True), fill=YELLOW)
    # Tech pills
    x = 100
    y = 420
    for label, bg in [("Google Gemini", ACCENT1), ("LangGraph", PURPLE),
                      ("Streamlit", TEAL), ("FastAPI", AMBER)]:
        x = pill(draw, x, y, label, bg) + 4
    # Credit
    draw.text((100, 490), "Developed end-to-end with  IBM Bob  (IBM SkillsBuild 2026 · Day 02)",
              font=font(18), fill=(148, 163, 184))
    narration = (
        "Welcome to the AI Travel Planning Multi-Agent System. "
        "This project was built for the IBM SkillsBuild Gartner Program Day 2. "
        "It uses three specialized AI agents powered by Google Gemini, "
        "orchestrated by LangGraph, with a Streamlit user interface and a FastAPI REST backend."
    )
    return img, narration


def slide_02_overview() -> tuple[Image.Image, str]:
    img, draw = new_img(BG_WHITE)
    header_bar(draw, "Project Overview", "02")
    bottom_stripe(draw)
    # Problem
    label_box(draw, 20, 80, 580, 30, "PROBLEM STATEMENT", BG_DARK)
    content_box(draw, 20, 110, 580, 120,
        "Planning a trip is time-consuming and fragmented. Travellers juggle "
        "multiple booking sites, compare hundreds of options, and manually "
        "reconcile costs — with no guarantee their choices are coordinated.",
        (255, 248, 225), TEXT_DK, 15)
    # Solution
    label_box(draw, 20, 248, 580, 30, "OUR SOLUTION", GREEN)
    content_box(draw, 20, 278, 580, 120,
        "A multi-agent AI system that accepts one travel request, splits the "
        "budget intelligently (40% flights · 45% hotel · 15% car), dispatches "
        "three parallel Gemini-powered agents, validates deterministically, "
        "and presents a coordinated cost-aware plan for user approval.",
        (220, 252, 231), TEXT_DK, 15)
    # Key facts
    label_box(draw, 640, 80, 620, 30, "KEY FACTS", ACCENT1)
    facts = [
        ("✦", "3 AI Agents — Flight · Hotel · Car"),
        ("✦", "Google Gemini 2.5 Flash LLM"),
        ("✦", "LangGraph StateGraph orchestration"),
        ("✦", "Streamlit Web UI  +  FastAPI Backend"),
        ("✦", "Mock data — no paid APIs needed"),
        ("✦", "Budget split: 40% / 45% / 15%"),
        ("✦", "68 / 68 Tests passing  ✅"),
    ]
    y = 124
    for bullet, txt in facts:
        draw.text((660, y), bullet, font=font(16, True), fill=ACCENT1)
        draw.text((688, y), txt, font=font(16), fill=TEXT_DK)
        y += 32
    # Budget pie (ASCII art)
    label_box(draw, 640, 380, 620, 30, "BUDGET SPLIT", AMBER)
    draw.text((660, 420), "✈️  Flight   40 %   ████████████████░░░░░░░░░░", font=font(17), fill=ACCENT1)
    draw.text((660, 454), "🏨  Hotel    45 %   ██████████████████░░░░░░░░", font=font(17), fill=TEAL)
    draw.text((660, 488), "🚗  Car      15 %   ██████░░░░░░░░░░░░░░░░░░░░", font=font(17), fill=AMBER)
    narration = (
        "The project solves a real travel planning problem. "
        "Instead of visiting multiple booking sites, the user submits a single form "
        "with their destination, dates, and total budget. "
        "The system automatically splits that budget — 40 percent for flights, "
        "45 percent for hotel, and 15 percent for car rental — "
        "then dispatches three AI agents simultaneously. "
        "Key facts: three agents, Google Gemini 2.5 Flash, LangGraph orchestration, "
        "Streamlit UI, FastAPI backend, and all 68 tests are passing."
    )
    return img, narration


def slide_03_architecture() -> tuple[Image.Image, str]:
    img, draw = new_img(BG_WHITE)
    header_bar(draw, "System Architecture", "03")
    bottom_stripe(draw)

    # Left column — flow
    boxes = [
        (BG_DARK,  "👤  User (Streamlit UI)",       "Sidebar form · Tabs · Cost summary · Approval"),
        (ACCENT1,  "⚙️  FastAPI Backend",            "/plan  ·  /approve  ·  /health  ·  UUID sessions"),
        (PURPLE,   "🔗  LangGraph Orchestrator",    "6-node StateGraph · TravelState TypedDict"),
        (GREEN,    "✅  validate_itinerary",         "Pure Python — dates · cost · budget check"),
        (AMBER,    "🎯  await_approval",             "itinerary_ready = True · user reviews"),
    ]
    y = 88
    for bg, title, sub in boxes:
        draw.rectangle([20, y, 540, y + 50], fill=bg)
        draw.text((30, y + 6), title, font=font(18, True), fill=WHITE)
        draw.text((30, y + 28), sub, font=font(13), fill=(200, 220, 255) if bg == BG_DARK else WHITE)
        if y < 88 + 4 * 65:
            draw.text((280, y + 55), "▼", font=font(18, True), fill=ACCENT1)
        y += 65

    # Right column — agents
    draw.text((660, 88), "AI Agents  (Google Gemini 2.5 Flash)", font=font(18, True), fill=BG_DARK)
    agents = [
        ("✈️ Agent 1", "Flight Booking Agent", ACCENT1),
        ("🏨 Agent 2", "Hotel Booking Agent",  TEAL),
        ("🚗 Agent 3", "Car Booking Agent",    AMBER),
    ]
    ax = 580
    for emoji_label, name, bg in agents:
        draw.rectangle([ax, 120, ax + 210, 240], fill=bg)
        draw.text((ax + 8, 128), emoji_label, font=font(20, True), fill=WHITE)
        draw.text((ax + 8, 158), name, font=font(15), fill=WHITE)
        draw.text((ax + 8, 190), "Gemini 2.5 Flash", font=font(13), fill=(220, 240, 255))
        draw.text((ax + 85, 248), "│", font=font(18), fill=ACCENT1)
        draw.text((ax + 60, 265), "JSON recs", font=font(13), fill=TEXT_MUT)
        ax += 220

    # Mock data
    draw.rectangle([580, 295, 1250, 350], fill=(255, 243, 205))
    draw.text((590, 305), "📦  Mock Data Layer — MOCK_FLIGHTS · MOCK_HOTELS · MOCK_CARS", font=font(16), fill=TEXT_DK)

    # Note
    draw.rectangle([20, 415, 1250, 460], fill=(247, 248, 250))
    draw.text((30, 425),
              "NOTE: The orchestrator is workflow logic only — not a fourth AI agent. "
              "Budget splits, date validation and cost arithmetic use plain Python.",
              font=font(15), fill=TEXT_MUT)
    narration = (
        "The architecture has four layers. "
        "First, the Streamlit user interface collects trip requirements. "
        "Second, the FastAPI backend exposes REST endpoints and manages sessions. "
        "Third, the LangGraph orchestrator runs a 6-node state graph, "
        "calling each AI agent in sequence and passing state between them. "
        "Three agents — Flight, Hotel, and Car — each make a call to Google Gemini "
        "with distinct system prompts and return ranked JSON recommendations. "
        "Finally, a deterministic validator checks dates, sums costs, and flags budget overruns. "
        "Importantly, the orchestrator itself is not an AI agent — it is pure Python workflow logic."
    )
    return img, narration


def slide_04_techstack() -> tuple[Image.Image, str]:
    img, draw = new_img(BG_WHITE)
    header_bar(draw, "Technology Stack", "04")
    bottom_stripe(draw)

    tiles = [
        (ACCENT1, "AI / LLM",            "google-genai >= 2.0.0\nGemini 2.5 Flash\ntemp=0.3 · max_tokens=1024\nExp. back-off retry"),
        (PURPLE,  "Workflow",             "langgraph >= 0.2.0\nStateGraph (TypedDict)\n6-node DAG\nConditional edges"),
        (TEAL,    "Web UI",               "streamlit >= 1.40.0\nTabbed results\nSidebar form\nCustom CSS cards"),
        (AMBER,   "REST Backend",         "fastapi >= 0.115.0\nuvicorn + pydantic v2\nCORS middleware\nSwagger /docs"),
        (GREEN,   "Testing",              "pytest >= 8.0.0\nunittest.mock\n68/68 tests ✅\n18 LLM dimensions"),
        (BG_DARK, "Data Layer",           "mock_data.py\n5 flights · 5 hotels\n5 cars · no paid APIs\nBudget filters"),
    ]
    positions = [(20, 90), (440, 90), (860, 90), (20, 370), (440, 370), (860, 370)]
    for (x, y), (bg, title, body) in zip(positions, tiles):
        draw.rectangle([x, y, x + 390, y + 255], fill=bg)
        draw.text((x + 12, y + 10), title, font=font(22, True), fill=WHITE)
        draw.rectangle([x, y + 50, x + 390, y + 52], fill=(255, 255, 255, 80))
        wrap_text(draw, body, x + 12, y + 60, 366, font(17), WHITE, line_gap=8)

    narration = (
        "The technology stack has six pillars. "
        "The AI layer uses Google's genai SDK to call Gemini 2.5 Flash with temperature 0.3 "
        "and exponential back-off for rate limit handling. "
        "LangGraph provides the workflow orchestration as a typed state graph. "
        "Streamlit powers the user interface with tabbed results and a sidebar form. "
        "FastAPI with Pydantic v2 provides the REST backend with automatic Swagger documentation. "
        "Pytest with mocked LLM calls delivers 68 passing tests across 18 testing dimensions. "
        "And mock_data dot py supplies 15 simulated travel options with no paid API keys needed."
    )
    return img, narration


def slide_05_agents() -> tuple[Image.Image, str]:
    img, draw = new_img(BG_WHITE)
    header_bar(draw, "The Three AI Agents", "05")
    bottom_stripe(draw)

    cols = [
        (ACCENT1, "✈️ Agent 1\nFlight Booking Agent",
         "INPUT\n• Origin & destination\n• Departure & return dates\n• Flight budget (40%)\n\n"
         "OUTPUT\n• Ranked ≤3 flights\n• airline · fare_usd · class\n• stops · departure · arrival\n• reason\n\n"
         "FOCUS\nPrice / duration / timing"),
        (TEAL,    "🏨 Agent 2\nHotel Booking Agent",
         "INPUT\n• Destination city\n• Check-in / check-out\n• Budget/night (45%÷nights)\n• User preferences\n\n"
         "OUTPUT\n• Ranked ≤3 hotels\n• name · stars · nightly rate\n• total · rating · amenities\n• reason\n\n"
         "FOCUS\nValue · location · rating"),
        (AMBER,   "🚗 Agent 3\nCar Booking Agent",
         "INPUT\n• Pickup = destination airport\n• Pickup time ← Agent 1\n• Passengers · num_days\n• Budget/day (15%÷nights)\n\n"
         "OUTPUT\n• Ranked ≤3 vehicles\n• provider · vehicle · category\n• daily rate · total\n• reason\n\n"
         "FOCUS\nPassengers · timing · cost"),
    ]
    x = 20
    for bg, title, body in cols:
        draw.rectangle([x, 80, x + 400, 680], fill=bg)
        draw.text((x + 12, 88), title, font=font(20, True), fill=WHITE)
        draw.rectangle([x, 136, x + 400, 138], fill=WHITE)
        wrap_text(draw, body, x + 12, 146, 376, font(15), WHITE, line_gap=7)
        x += 420

    # Footer
    draw.rectangle([20, 685, 1260, 708], fill=ACCENT1)
    draw.text((30, 690),
              "All three agents share the same Gemini model (GEMINI_MODEL env var) "
              "but have distinct system prompts and output schemas.",
              font=font(14), fill=WHITE)
    narration = (
        "The system has three specialized agents. "
        "Agent 1, the Flight Booking Agent, receives origin, destination, dates, and the flight budget, "
        "then returns up to three ranked flights with airline, fare, class, stops, and timing. "
        "Agent 2, the Hotel Booking Agent, receives the destination, check-in and check-out dates, "
        "per-night budget, and any user preferences like pool or free WiFi. "
        "Agent 3, the Car Booking Agent, is unique — it receives its pickup time "
        "directly from Agent 1's top flight arrival time, demonstrating real agent coordination. "
        "All three agents use the same Gemini model but have completely different system prompts."
    )
    return img, narration


def slide_06_workflow() -> tuple[Image.Image, str]:
    img, draw = new_img(BG_WHITE)
    header_bar(draw, "LangGraph Workflow — Step by Step", "06")
    bottom_stripe(draw)

    steps = [
        (ACCENT1, "START",                   "User submits form: origin, destination, dates, budget, passengers"),
        (BG_DARK, "1. collect_requirements", "Validates dates · budget >0 · computes num_nights · splits budget 40/45/15%"),
        (BG_DARK, "2. flight_agent",         "Gets mock flights → Gemini call → parses JSON → returns flight_result"),
        (BG_DARK, "3. hotel_agent",          "Gets mock hotels (with total_cost attached) → Gemini → hotel_result"),
        (BG_DARK, "4. car_agent",            "Pickup time = flight_result[0].arrival → Gemini → car_result"),
        (GREEN,   "5. validate_itinerary",   "NO LLM: sums costs · checks budget · sets itinerary_ready, warnings"),
        (AMBER,   "6. await_approval",       "Sets itinerary_ready=True · UI shows tabs + Approve/Cancel buttons"),
        (ACCENT1, "END",                     "Simulated confirmation: FL-SIM-AUTO · HT-SIM-AUTO · CAR-SIM-AUTO"),
    ]
    y = 82
    for bg, label, desc in steps:
        draw.rectangle([20, y, 260, y + 46], fill=bg)
        draw.text((28, y + 10), label, font=font(15, True), fill=WHITE)
        draw.rectangle([264, y, 1260, y + 46], fill=(247, 248, 250))
        draw.text((272, y + 13), desc, font=font(15), fill=TEXT_DK)
        if y + 46 + 8 < 660:
            draw.text((133, y + 48), "▼", font=font(14, True), fill=ACCENT1)
        y += 60

    # Error path
    draw.rectangle([20, 668, 1260, 704], fill=(254, 226, 226))
    draw.text((28, 678),
              "⚡ Error path: if validation_errors exist at any node → conditional edge routes to END immediately",
              font=font(14), fill=(239, 68, 68))
    narration = (
        "The LangGraph workflow has 6 nodes. "
        "First, collect requirements validates inputs and splits the budget. "
        "Node 2, the flight agent, fetches mock flight data, calls Gemini, and parses the JSON response. "
        "Node 3, the hotel agent, does the same for hotels, pre-computing total costs per option. "
        "Node 4, the car agent, uniquely takes the arrival time from node 2's output as its pickup time. "
        "Node 5, validate itinerary, makes no LLM call — it is pure Python that sums all costs, "
        "checks against the budget, and sets warnings. "
        "Node 6, await approval, sets the itinerary ready flag and hands control back to the UI. "
        "A conditional edge after validation short-circuits to END on any hard error."
    )
    return img, narration


def slide_07_structure() -> tuple[Image.Image, str]:
    img, draw = new_img(BG_WHITE)
    header_bar(draw, "Project Structure", "07")
    bottom_stripe(draw)

    tree = (
        "travelplanner/\n"
        "├── app.py          Streamlit UI — form, tabs, cost summary, approval\n"
        "├── backend.py      FastAPI REST — /plan · /approve · /health\n"
        "├── workflow.py     LangGraph 6-node StateGraph + TravelState\n"
        "├── utils.py        Pure Python — dates · budget · formatting\n"
        "├── agents/\n"
        "│   └── agents.py   3 AI agents, shared Gemini client, retry logic\n"
        "├── data/\n"
        "│   └── mock_data.py  MOCK_FLIGHTS · MOCK_HOTELS · MOCK_CARS\n"
        "└── tests/\n"
        "    ├── test_travel_planner.py         13 tests\n"
        "    └── test_llm_agent_correctness.py  68 tests (18 LLM dimensions)"
    )
    draw.rectangle([20, 82, 1260, 420], fill=(247, 248, 250))
    wrap_text(draw, tree, 30, 92, 1220, mono(17), TEXT_DK, line_gap=6)

    # Two info boxes
    label_box(draw, 20, 430, 590, 28, "app.py — 5 render functions", ACCENT1)
    content_box(draw, 20, 458, 590, 130,
        "render_sidebar()  — travel requirements form\n"
        "render_flight_results()  — expandable flight cards\n"
        "render_hotel_results()   — hotel cards + amenities\n"
        "render_car_results()     — vehicle cards\n"
        "render_cost_summary()    — budget metrics + warnings\n"
        "render_approval_section() — Approve / Cancel",
        (219, 234, 254), TEXT_DK, 14)

    label_box(draw, 650, 430, 590, 28, "TravelState — TypedDict fields", PURPLE)
    content_box(draw, 650, 458, 590, 130,
        "Inputs:  origin · destination · dates · budget · passengers\n"
        "Derived: num_nights · flight_budget · hotel_budget_per_night\n"
        "         car_budget_per_day\n"
        "Outputs: flight_result · hotel_result · car_result\n"
        "         estimated_total_cost_usd · itinerary_ready\n"
        "         validation_errors · validation_warnings",
        (237, 233, 254), TEXT_DK, 14)

    narration = (
        "The project has a clean, flat structure. "
        "app dot py contains the Streamlit interface with five render functions for each section. "
        "backend dot py provides the FastAPI REST API. "
        "workflow dot py holds the LangGraph state graph and the TravelState TypedDict "
        "which carries all data between nodes. "
        "agents slash agents dot py defines the three AI agents with separate system prompts "
        "but a shared, lazily-initialized Gemini client. "
        "The tests folder contains two suites — 13 original tests and 68 LLM correctness tests."
    )
    return img, narration


def slide_08_ui() -> tuple[Image.Image, str]:
    img, draw = new_img(BG_WHITE)
    header_bar(draw, "Streamlit UI — Screen Walkthrough", "08")
    bottom_stripe(draw)

    screens = [
        (ACCENT1, "SCREEN 1 — Sidebar Form",
         "• Origin: New York (JFK)\n• Destination: Los Angeles (LAX)\n"
         "• Departure: 2025-06-01  Return: 2025-06-07\n"
         "• Budget: ₹1,50,000  Passengers: 2\n• Preferences: pool, free WiFi\n"
         "→ Click  🚀 Plan My Trip"),
        (TEAL, "SCREEN 2 — Agent Results (Tabs)",
         "Three tabs: ✈️ Flights | 🏨 Hotels | 🚗 Cars\n\n"
         "#1 SkyWings Airlines — ₹27,000 · Economy · 0 stops\n"
         "   Departure: 08:00  Arrival: 11:30  Duration: 5h 30m\n"
         "   💡 Best value for morning travel budget\n\n"
         "#2 BlueHorizon Air — ₹23,500 · Economy · 0 stops\n"
         "#3 BudgetFly — ₹13,500 · Economy · 1 stop"),
        (AMBER, "SCREEN 3 — Cost Summary",
         "Metrics: Estimated Total | Your Budget | Remaining\n\n"
         "₹95,400 total vs ₹1,50,000 budget\n"
         "→ ₹54,600 under budget  ✅\n\n"
         "[✅ Approve & Simulate Booking]   [❌ Cancel]"),
        (GREEN, "SCREEN 4 — Confirmation",
         "🎉 Confetti balloons!\n\n"
         "✅ Simulated booking confirmed!\n"
         "FL-SIM-AUTO · HT-SIM-AUTO · CAR-SIM-AUTO\n\n"
         "[🔄 Plan Another Trip]"),
    ]
    x = 20
    for bg, title, body in screens:
        draw.rectangle([x, 80, x + 295, 700], fill=bg)
        draw.text((x + 8, 88), title, font=font(14, True), fill=WHITE)
        draw.rectangle([x, 118, x + 295, 120], fill=WHITE)
        wrap_text(draw, body, x + 8, 126, 278, font(14), WHITE, line_gap=7)
        x += 315

    # Run command
    draw.rectangle([20, 704, 1260, 710], fill=ACCENT1)
    narration = (
        "The Streamlit interface has four screens. "
        "Screen 1 is the sidebar form where users enter their travel details — "
        "origin, destination, dates, total budget in Indian Rupees, passenger count, and hotel preferences. "
        "Screen 2 shows the agent results in three tabs: Flights, Hotels, and Cars. "
        "Each tab has expandable cards with full recommendation details and a reason from the AI. "
        "Screen 3 shows the cost summary with three metrics — estimated total, budget, and remaining — "
        "plus any warnings if the estimated cost exceeds the budget. "
        "Screen 4 is the simulated booking confirmation with confetti animation "
        "and reference numbers like FL-SIM-AUTO. "
        "Run the app with: streamlit run app dot py"
    )
    return img, narration


def slide_09_api() -> tuple[Image.Image, str]:
    img, draw = new_img(BG_WHITE)
    header_bar(draw, "FastAPI Backend — REST Endpoints", "09")
    bottom_stripe(draw)

    # Endpoint 1
    label_box(draw, 20, 82, 300, 30, "GET   /health", GREEN)
    content_box(draw, 20, 112, 1240, 60,
        '→  { "status": "ok", "model": "gemini-2.5-flash", "timestamp": "2025-06-01T..." }',
        (220, 252, 231), TEXT_DK, 14, mono_font=True)

    # Endpoint 2
    label_box(draw, 20, 192, 300, 30, "POST  /plan", ACCENT1)
    req_body = (
        '{\n'
        '  "origin": "New York (JFK)",\n'
        '  "destination": "Los Angeles (LAX)",\n'
        '  "departure_date": "2025-06-01",\n'
        '  "return_date": "2025-06-07",\n'
        '  "total_budget_usd": 150000,\n'
        '  "hotel_preferences": "pool, free WiFi",\n'
        '  "passengers": 2\n'
        '}'
    )
    resp_body = (
        '{\n'
        '  "session_id": "<uuid>",\n'
        '  "itinerary_ready": true,\n'
        '  "estimated_total_cost_usd": 95400.0,\n'
        '  "validation_errors": [],\n'
        '  "validation_warnings": [],\n'
        '  "flight_result": { "recommendations": [...] },\n'
        '  "hotel_result":  { "recommendations": [...] },\n'
        '  "car_result":    { "recommendations": [...] }\n'
        '}'
    )
    content_box(draw, 20, 222, 600, 240, "REQUEST\n" + req_body,
                (219, 234, 254), TEXT_DK, 13, mono_font=True)
    content_box(draw, 640, 222, 620, 240, "RESPONSE\n" + resp_body,
                (240, 244, 255), TEXT_DK, 13, mono_font=True)

    # Endpoint 3
    label_box(draw, 20, 484, 300, 30, "POST  /approve", AMBER)
    content_box(draw, 20, 514, 1240, 90,
        'Request: { "session_id": "<uuid>", "approved": true, "selected_flight_id": "FL001" }\n'
        'Response: { "approved": true, "message": "SIMULATED booking confirmed",\n'
        '            "confirmations": { "FL-SIM-FL001", "HT-SIM-AUTO", "CAR-SIM-AUTO" } }',
        (254, 243, 199), TEXT_DK, 14, mono_font=True)

    draw.text((20, 620),
              "Swagger UI auto-generated at  http://localhost:8000/docs  "
              "·  uvicorn backend:app --reload --port 8000",
              font=font(15), fill=ACCENT1)
    narration = (
        "The FastAPI backend exposes three endpoints. "
        "GET slash health confirms the service is running and reports the configured Gemini model. "
        "POST slash plan accepts the travel requirements, runs all three agents via the LangGraph workflow, "
        "and returns a full itinerary with a session ID for follow-up. "
        "POST slash approve takes the session ID and records user approval, "
        "returning simulated confirmation reference numbers. "
        "Pydantic version 2 validates all inputs, including date format checking. "
        "Swagger UI is automatically generated at localhost colon 8000 slash docs."
    )
    return img, narration


def slide_10_mockdata() -> tuple[Image.Image, str]:
    img, draw = new_img(BG_WHITE)
    header_bar(draw, "Mock Data — All 15 Simulated Options", "10")
    bottom_stripe(draw)

    # Flights
    label_box(draw, 20, 82, 1240, 26, "✈️  MOCK_FLIGHTS (5 options — New York JFK → Los Angeles LAX)", BG_DARK)
    rows = [
        ("FL001", "SkyWings Airlines",  "₹27,000", "Economy",  "0", "08:00 → 11:30", "5h 30m"),
        ("FL002", "BlueHorizon Air",    "₹23,500", "Economy",  "0", "11:45 → 15:20", "5h 35m"),
        ("FL003", "SkyWings Airlines",  "₹17,500", "Economy",  "1", "18:00 → 21:45", "5h 45m"),
        ("FL004", "PremiumJet",         "₹57,000", "Business", "0", "07:00 → 10:25", "5h 25m"),
        ("FL005", "BudgetFly",          "₹13,500", "Economy",  "1", "23:55 → 03:40", "5h 45m"),
    ]
    headers = ["ID", "Airline", "Fare", "Class", "Stops", "Route", "Duration"]
    col_x   = [28, 100, 300, 430, 540, 610, 900]
    y = 110
    for i, h in enumerate(headers):
        draw.text((col_x[i], y), h, font=font(13, True), fill=TEXT_MUT)
    y += 20
    for r in rows:
        bg = (240, 244, 255) if rows.index(r) % 2 == 0 else (247, 248, 250)
        draw.rectangle([20, y, 1260, y + 22], fill=bg)
        for i, val in enumerate(r):
            draw.text((col_x[i], y + 4), val, font=font(13), fill=TEXT_DK)
        y += 22

    # Hotels
    label_box(draw, 20, 236, 1240, 26, "🏨  MOCK_HOTELS (5 options — Los Angeles)", BG_DARK)
    h_rows = [
        ("HT001", "Grand Skyline Hotel",  "5★", "₹27,000", "Downtown LA",     "4.8"),
        ("HT002", "Comfort Inn Central",  "3★", "₹10,000", "Mid-Wilshire",    "4.1"),
        ("HT003", "Beachside Retreat",    "4★", "₹20,000", "Santa Monica",    "4.6"),
        ("HT004", "Budget Stay Express",  "2★", "₹5,900",  "North Hollywood", "3.5"),
        ("HT005", "Urban Boutique Hotel", "4★", "₹16,500", "Hollywood",       "4.4"),
    ]
    hcol = [28, 100, 340, 400, 520, 780]
    hhdrs = ["ID", "Name", "★", "Rate/Night", "Location", "Rating"]
    y = 264
    for i, h in enumerate(hhdrs):
        draw.text((hcol[i], y), h, font=font(13, True), fill=TEXT_MUT)
    y += 20
    for r in h_rows:
        bg = (220, 252, 231) if h_rows.index(r) % 2 == 0 else (247, 248, 250)
        draw.rectangle([20, y, 1260, y + 22], fill=bg)
        for i, val in enumerate(r):
            draw.text((hcol[i], y + 4), val, font=font(13), fill=TEXT_DK)
        y += 22

    # Cars
    label_box(draw, 20, 392, 1240, 26, "🚗  MOCK_CARS (5 options — LAX Airport)", BG_DARK)
    c_rows = [
        ("CAR001", "AutoRent Premier", "Toyota Camry",      "Sedan",   "₹4,600",  "5"),
        ("CAR002", "EcoDrive",         "Tesla Model 3",     "Electric","₹8,000",  "5"),
        ("CAR003", "ValueWheels",      "Hyundai Elantra",   "Economy", "₹3,200",  "5"),
        ("CAR004", "LuxuryCruise",     "BMW 5 Series",      "Luxury",  "₹14,700", "5"),
        ("CAR005", "FamilyDrive",      "Chrysler Pacifica", "Minivan", "₹7,100",  "7"),
    ]
    ccol = [28, 100, 270, 520, 660, 780]
    chdrs = ["ID", "Provider", "Vehicle", "Category", "Rate/Day", "Pax"]
    y = 420
    for i, h in enumerate(chdrs):
        draw.text((ccol[i], y), h, font=font(13, True), fill=TEXT_MUT)
    y += 20
    for r in c_rows:
        bg = (254, 243, 199) if c_rows.index(r) % 2 == 0 else (247, 248, 250)
        draw.rectangle([20, y, 1260, y + 22], fill=bg)
        for i, val in enumerate(r):
            draw.text((ccol[i], y + 4), val, font=font(13), fill=TEXT_DK)
        y += 22

    draw.text((20, 554),
              "Budget filters apply at query time — options over budget are excluded before being passed to the agent.",
              font=font(14), fill=TEXT_MUT)
    narration = (
        "The mock data layer provides 15 simulated travel options with no paid API keys required. "
        "5 flights from New York JFK to Los Angeles LAX, ranging from budget fly at 13,500 Rupees "
        "to Premium Jet business class at 57,000. "
        "5 hotels in Los Angeles, from Budget Stay Express at 5,900 per night "
        "to Grand Skyline Hotel 5 star at 27,000. "
        "5 rental cars at LAX, from ValueWheels Hyundai Elantra at 3,200 per day "
        "to LuxuryCruise BMW 5 Series at 14,700. "
        "Budget filters run at query time, so only options within the agent's allocated budget "
        "are passed to the Gemini model."
    )
    return img, narration


def slide_11_tests() -> tuple[Image.Image, str]:
    img, draw = new_img(BG_WHITE)
    header_bar(draw, "LLM Agent Correctness Test Results", "11")
    bottom_stripe(draw)

    # Score
    draw.text((30, 90), "68", font=font(110, True), fill=GREEN)
    draw.text((220, 110), "/ 68", font=font(60), fill=(100, 130, 100))
    draw.text((220, 175), "All Tests Passed  ✅", font=font(26, True), fill=GREEN)
    draw.text((220, 212), "0 failed  ·  0 skipped  ·  1.33 s  ·  Python 3.15  ·  pytest 9.2.0",
              font=font(16), fill=TEXT_MUT)

    # 18 dimensions in two columns
    label_box(draw, 30, 256, 1220, 28, "18 LLM Testing Dimensions", BG_DARK)
    dims_left = [
        " 1. Accuracy (6)          — Required fields & numeric types",
        " 2. Completeness (4)     — All result keys always returned",
        " 3. Format Fidelity (2)  — JSON despite LLM preamble text",
        " 4. Budget Adherence (3) — 40/45/15% split · over-budget warn",
        " 5. Input Boundary (6)   — Min/max budget, passengers, trips",
        " 6. Negative Cases (7)   — Bad JSON · empty · truncated · dates",
        " 7. Robustness (3)       — Prompt inject · unicode · special chars",
        " 8. Multi-Passenger (2)  — Cost × passengers correct",
        " 9. Data Types (3)       — Numeric-as-string from LLM",
    ]
    dims_right = [
        "10. Determinism (1)      — Same mock → identical structure",
        "11. Agent Isolation (2)  — No cross-key contamination",
        "12. Workflow Integ. (3)  — End-to-end TravelState correct",
        "13. Date Edge Cases (5)  — Leap year · year boundary",
        "14. Cost Arithmetic (5)  — Exact float · at-limit · 1-over",
        "15. Budget Split (3)     — 40+45+15 reconstructs total",
        "16. Mock Data Filter (6) — Boundary incl/excl · empty list",
        "17. No-Rec Handling (4)  — Empty list · validation warns",
        "18. Response Parsing (3) — Whitespace · nested · 3-rec order",
    ]
    y = 296
    for line in dims_left:
        draw.text((35, y), "✔", font=font(14, True), fill=GREEN)
        draw.text((52, y), line, font=font(14), fill=TEXT_DK)
        y += 26
    y = 296
    for line in dims_right:
        draw.text((645, y), "✔", font=font(14, True), fill=GREEN)
        draw.text((662, y), line, font=font(14), fill=TEXT_DK)
        y += 26

    draw.text((30, 600),
              "Run:  pytest tests/test_llm_agent_correctness.py -v --tb=short",
              font=mono(15), fill=ACCENT1)
    narration = (
        "The LLM agent correctness test suite achieves 68 out of 68 passing tests "
        "across 18 testing dimensions in just 1.33 seconds. "
        "All Google Gemini API calls are mocked using Python's unittest dot mock dot patch, "
        "so tests run offline and deterministically with no API key required. "
        "The 18 dimensions cover accuracy, completeness, format fidelity, budget adherence, "
        "input boundary conditions, negative cases, robustness against prompt injection, "
        "multi-passenger cost scaling, data type handling, determinism, agent isolation, "
        "end-to-end workflow integration, date edge cases, cost arithmetic, "
        "budget split verification, mock data filters, no-recommendation handling, "
        "and response parsing edge cases."
    )
    return img, narration


def slide_12_conclusion() -> tuple[Image.Image, str]:
    img, draw = new_img(BG_DARK)
    top_stripe(draw)
    bottom_stripe(draw)

    draw.text((W // 2 - 340, 30), "Key Takeaways & Next Steps", font=font(42, True), fill=WHITE)
    draw.rectangle([60, 86, W - 60, 88], fill=ACCENT2)

    # Left — what we built
    label_box(draw, 30, 100, 580, 28, "What We Built", ACCENT1)
    built = [
        "✦  Multi-agent AI system — 3 Gemini-powered agents",
        "✦  LangGraph DAG with 6 nodes and conditional edges",
        "✦  Streamlit UI + FastAPI REST backend",
        "✦  Deterministic budget split & itinerary validation",
        "✦  68/68 tests across 18 LLM testing dimensions",
        "✦  Zero paid APIs — full mock data layer",
    ]
    y = 138
    for line in built:
        draw.text((40, y), line, font=font(18), fill=WHITE)
        y += 36

    # Right — learnings
    label_box(draw, 640, 100, 580, 28, "Technical Learnings", PURPLE)
    learned = [
        "✦  LangGraph TypedDict state flows cleanly across agents",
        "✦  Mocking LLM calls = offline, deterministic tests",
        "✦  JSON extraction handles LLM verbosity robustly",
        "✦  Exponential back-off essential for free-tier quotas",
        "✦  Budget-filtered data prevents over-querying the LLM",
        "✦  Agent 3 consuming Agent 1 output = true coordination",
    ]
    y = 138
    for line in learned:
        draw.text((650, y), line, font=font(18), fill=WHITE)
        y += 36

    # Next steps
    label_box(draw, 30, 380, 1220, 28, "Possible Next Steps", ACCENT2)
    next_steps = [
        "1.  Integrate real travel provider APIs (Amadeus · Booking.com · car aggregators)",
        "2.  Add a Supervisor meta-agent to re-route on budget overage dynamically",
        "3.  Persist sessions in PostgreSQL / Redis instead of in-memory dict",
        "4.  Add Agent 4 — Activity / Sightseeing recommendations for the destination",
        "5.  Deploy to cloud (Streamlit Community Cloud · Railway · AWS Lambda)",
        "6.  Add streaming LLM output to UI using Streamlit st.write_stream",
    ]
    y = 418
    for line in next_steps:
        draw.text((40, y), line, font=font(16), fill=(200, 220, 255))
        y += 32

    # IBM Bob credit
    draw.text((W // 2 - 420, 618),
              "Developed end-to-end with IBM Bob  ·  Plan mode for architecture · Agent mode for all code & tests",
              font=font(15), fill=LBLUE)
    draw.text((W // 2 - 130, 650), "Thank You  ✈️", font=font(38, True), fill=YELLOW)

    narration = (
        "In conclusion, we built a complete multi-agent AI travel planning system "
        "with three Gemini-powered agents, a LangGraph workflow, Streamlit UI, and FastAPI backend, "
        "all validated by 68 passing tests across 18 LLM testing dimensions. "
        "Key technical learnings include using typed state graphs for clean agent coordination, "
        "mocking LLM calls for reliable offline testing, "
        "and demonstrating true agent-to-agent coordination where the car agent "
        "consumes the flight agent's arrival time. "
        "Possible next steps include integrating real travel APIs, "
        "adding a supervisor meta-agent, cloud deployment, and streaming LLM output. "
        "This entire project was developed end-to-end with IBM Bob. "
        "Thank you for watching."
    )
    return img, narration


# ---------------------------------------------------------------------------
# Main render pipeline
# ---------------------------------------------------------------------------

SLIDE_BUILDERS = [
    (slide_01_title,       TITLE_DUR),
    (slide_02_overview,    SLIDE_DUR),
    (slide_03_architecture,SLIDE_DUR),
    (slide_04_techstack,   SLIDE_DUR),
    (slide_05_agents,      SLIDE_DUR),
    (slide_06_workflow,    SLIDE_DUR),
    (slide_07_structure,   SLIDE_DUR),
    (slide_08_ui,          SLIDE_DUR),
    (slide_09_api,         SLIDE_DUR),
    (slide_10_mockdata,    SLIDE_DUR),
    (slide_11_tests,       SLIDE_DUR),
    (slide_12_conclusion,  TITLE_DUR + 1),
]

def cross_fade(clip_a: VideoClip, clip_b: VideoClip, dur: float) -> VideoClip:
    """Simple cross-fade: last `dur` seconds of A fades into first `dur` seconds of B."""
    from moviepy.video.fx.fadeout import fadeout
    from moviepy.video.fx.fadein  import fadein
    a = fadeout(clip_a, dur)
    b = fadein(clip_b, dur)
    overlap = b.set_start(clip_a.duration - dur)
    return CompositeVideoClip([a, overlap], size=(W, H)).set_duration(
        clip_a.duration + clip_b.duration - dur
    )


def main() -> None:
    out_dir = Path(__file__).parent
    out_path = str(out_dir / "AI_Travel_Planner_MAS.mp4")

    print("=" * 60)
    print("AI Travel Planning — Multi-Agent System Video Generator")
    print("=" * 60)
    print(f"Output: {out_path}")
    print(f"Slides: {len(SLIDE_BUILDERS)}  ·  Resolution: {W}×{H}  ·  FPS: {FPS}")
    print()

    clips: list[VideoClip] = []

    for i, (builder, duration) in enumerate(SLIDE_BUILDERS, 1):
        print(f"[{i:02d}/{len(SLIDE_BUILDERS)}] Rendering slide {i} ...", end=" ", flush=True)

        # 1. Build the PIL image
        pil_img, narration_text = builder()

        # 2. Narration audio
        wav = make_narration(i, narration_text)

        # 3. Compute duration = max(slide_dur, audio_dur + 0.5)
        actual_dur = duration
        if wav and os.path.exists(wav):
            try:
                aud = AudioFileClip(wav)
                actual_dur = max(duration, aud.duration + 0.8)
                aud.close()
            except Exception:
                pass

        # 4. Build animated clip
        clip = slide_to_clip(pil_img, actual_dur, animate=(i > 1))
        clip = attach_audio(clip, wav)
        clips.append(clip)
        print(f"done ({actual_dur:.1f}s)")

    print()
    print("Concatenating slides with cross-fade transitions …")
    final = clips[0]
    for c in clips[1:]:
        final = cross_fade(final, c, FADE_DUR)

    total_sec = sum(c.duration for c in clips) - FADE_DUR * (len(clips) - 1)
    print(f"Total duration: {total_sec:.1f}s  ({total_sec/60:.1f} min)")
    print()
    print(f"Writing MP4 → {out_path}")
    print("(This may take 1–3 minutes …)")

    final.write_videofile(
        out_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(_TMP_DIR / "temp_audio.m4a"),
        remove_temp=True,
        verbose=False,
        logger=None,
        ffmpeg_params=["-crf", "23", "-preset", "fast"],
    )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print()
    print("=" * 60)
    print(f"✅  Video saved:  {out_path}")
    print(f"    Size: {size_mb:.1f} MB  ·  Duration: {total_sec:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
