"""Streamlit Frontend for World Cup 2026 Prediction Engine."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="🏆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

ALL_TEAMS = sorted(
    [
        "Albania",
        "Algeria",
        "Argentina",
        "Australia",
        "Austria",
        "Bahrain",
        "Belgium",
        "Bolivia",
        "Bosnia and Herzegovina",
        "Brazil",
        "Burkina Faso",
        "Cameroon",
        "Canada",
        "Cape Verde",
        "Chile",
        "China PR",
        "Colombia",
        "Costa Rica",
        "Croatia",
        "Curaçao",
        "Czech Republic",
        "DR Congo",
        "Denmark",
        "Ecuador",
        "Egypt",
        "El Salvador",
        "England",
        "Finland",
        "France",
        "Georgia",
        "Germany",
        "Ghana",
        "Greece",
        "Guatemala",
        "Haiti",
        "Honduras",
        "Hungary",
        "Iceland",
        "Iran",
        "Iraq",
        "Italy",
        "Ivory Coast",
        "Jamaica",
        "Japan",
        "Jordan",
        "Mali",
        "Mexico",
        "Morocco",
        "Netherlands",
        "New Zealand",
        "Nigeria",
        "Northern Ireland",
        "Norway",
        "Oman",
        "Panama",
        "Paraguay",
        "Peru",
        "Poland",
        "Portugal",
        "Qatar",
        "Republic of Ireland",
        "Romania",
        "Saudi Arabia",
        "Scotland",
        "Senegal",
        "Serbia",
        "Slovakia",
        "Slovenia",
        "South Africa",
        "South Korea",
        "Spain",
        "Sweden",
        "Switzerland",
        "Syria",
        "Tunisia",
        "Turkey",
        "UAE",
        "Ukraine",
        "United States",
        "Uruguay",
        "Uzbekistan",
        "Venezuela",
        "Wales",
        "Zambia",
    ]
)


def get_prediction(
    home_team: str,
    away_team: str,
    tournament: str,
    neutral: bool,
) -> dict[str, Any] | None:
    """Call the FastAPI backend and return the prediction payload."""
    payload = {
        "home_team": home_team,
        "away_team": away_team,
        "tournament": tournament,
        "neutral": neutral,
    }
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ **Cannot reach the Prediction API** at `{API_URL}`.\n\n"
            "Make sure the backend is running:\n"
            "```bash\n"
            "uv run uvicorn backend.api.main:app --host 0.0.0.0 --port 8000\n"
            "```"
        )
    except requests.exceptions.Timeout:
        st.error("⏱️ The API took too long to respond. Please try again.")
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:  # noqa: BLE001
            detail = str(exc)
        st.error(f"❌ **API Error {exc.response.status_code}**: {detail}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Unexpected error: {exc}")
    return None


# ── Theme state ───────────────────────────────────────────────────────────────
# Architecture:
#   - config.toml base="dark" so Streamlit native widgets default to dark.
#   - When the user flips to light, we inject comprehensive CSS that overrides
#     every Streamlit internal selector back to a clean light palette.
#   - Dark mode: we only inject CSS for our *custom* HTML components since
#     Streamlit handles the base dark styles itself via config.toml.
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    new_dark: bool = st.toggle("🌙 Dark Mode", value=st.session_state["dark_mode"])
    if new_dark != st.session_state["dark_mode"]:
        st.session_state["dark_mode"] = new_dark
    st.divider()

    st.divider()
    st.caption("v0.1.0-alpha | Supabase Migrated")

dark_mode: bool = st.session_state["dark_mode"]

# ── Design tokens ─────────────────────────────────────────────────────────────
# Both palettes are fully specified so either branch is self-contained.
if dark_mode:
    # GitHub-dark inspired — deep navy blacks, electric blue accent
    PALETTE = {
        "bg_primary": "#0d1117",
        "bg_secondary": "#161b22",
        "bg_card": "rgba(22, 27, 34, 0.85)",
        "border": "rgba(48, 54, 61, 0.8)",
        "text_primary": "#e6edf3",
        "text_secondary": "#8b949e",
        "accent": "#3b9eff",
        "accent_glow": "rgba(59, 158, 255, 0.25)",
        "win_color": "#3b9eff",
        "draw_color": "#6e7681",
        "loss_color": "#f85149",
        "banner_bg": "linear-gradient(135deg, rgba(22,27,34,0.9) 0%, rgba(13,17,23,0.95) 100%)",
        "banner_border": "rgba(59, 158, 255, 0.35)",
        "banner_shadow": "0 8px 32px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(59, 158, 255, 0.15)",
        "metric_bg": "rgba(22, 27, 34, 0.6)",
        "bar_chart_bg": "rgba(0,0,0,0)",
        "plotly_paper": "rgba(0,0,0,0)",
    }
else:
    # Warm off-white — premium editorial, clean sports analytics feel
    PALETTE = {
        "bg_primary": "#f6f8fa",
        "bg_secondary": "#ffffff",
        "bg_card": "rgba(255, 255, 255, 0.95)",
        "border": "rgba(208, 215, 222, 0.8)",
        "text_primary": "#24292f",
        "text_secondary": "#57606a",
        "accent": "#0969da",
        "accent_glow": "rgba(9, 105, 218, 0.15)",
        "win_color": "#0969da",
        "draw_color": "#656d76",
        "loss_color": "#d1242f",
        "banner_bg": "linear-gradient(135deg, #0969da 0%, #218bff 100%)",
        "banner_border": "transparent",
        "banner_shadow": "0 8px 24px rgba(9, 105, 218, 0.3)",
        "metric_bg": "rgba(246, 248, 250, 0.9)",
        "bar_chart_bg": "rgba(0,0,0,0)",
        "plotly_paper": "rgba(0,0,0,0)",
    }

P = PALETTE

# ── Comprehensive CSS injection ───────────────────────────────────────────────
# Dark mode: only custom component styles (Streamlit native handles the rest).
# Light mode: full override of ALL Streamlit internals since config.toml is dark.
LIGHT_OVERRIDE_CSS = (
    """
    /* ── Full Streamlit light-mode override ──────────────────────────────── */
    /* Main app shell */
    .stApp,
    .stApp > header,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stMain"] {{
        background-color: {bg_primary} !important;
        color: {text_primary} !important;
    }}

    /* Main content block */
    [data-testid="stMainBlockContainer"],
    [data-testid="block-container"],
    .block-container {{
        background-color: {bg_primary} !important;
        color: {text_primary} !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    .css-1d391kg, .css-sidebar {{
        background-color: {bg_secondary} !important;
        color: {text_primary} !important;
        border-right: 1px solid {border} !important;
    }}

    /* All markdown / text */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2,
    .stMarkdown h3, .stMarkdown h4, p, h1, h2, h3, label,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stText"] {{
        color: {text_primary} !important;
    }}

    /* Caption / small text */
    .stCaption, [data-testid="stCaptionContainer"],
    .stCaption p {{
        color: {text_secondary} !important;
    }}

    /* Selectbox, inputs */
    [data-testid="stSelectbox"] > div > div,
    [data-baseweb="select"] > div,
    .stSelectbox div[data-baseweb="select"] > div,
    [data-baseweb="input"] input,
    input, select, textarea {{
        background-color: {bg_secondary} !important;
        color: {text_primary} !important;
        border-color: {border} !important;
    }}

    /* Dropdown popup items */
    [data-baseweb="popover"] ul,
    [data-baseweb="popover"] li,
    [data-baseweb="menu"] ul,
    [data-baseweb="menu"] li {{
        background-color: {bg_secondary} !important;
        color: {text_primary} !important;
    }}
    [data-baseweb="menu"] li:hover {{
        background-color: {bg_primary} !important;
    }}

    /* Checkbox */
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] span {{
        color: {text_primary} !important;
    }}

    /* Toggle */
    [data-testid="stToggle"] label,
    [data-testid="stToggle"] p {{
        color: {text_primary} !important;
    }}

    /* Metric */
    [data-testid="stMetric"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="metric-container"] {{
        background-color: {metric_bg} !important;
        color: {text_primary} !important;
        border: 1px solid {border} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"] {{
        color: {text_primary} !important;
    }}

    /* Expander */
    [data-testid="stExpander"],
    [data-testid="stExpanderDetails"] {{
        background-color: {bg_secondary} !important;
        border: 1px solid {border} !important;
        border-radius: 12px !important;
        color: {text_primary} !important;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {{
        color: {text_primary} !important;
    }}

    /* Info / alert boxes */
    [data-testid="stAlert"],
    .stAlert {{
        background-color: {metric_bg} !important;
        color: {text_primary} !important;
        border-color: {border} !important;
    }}

    /* Divider */
    hr {{
        border-color: {border} !important;
        opacity: 0.6;
    }}

    /* Spinner */
    .stSpinner > div {{
        border-top-color: {accent} !important;
    }}

    /* Code blocks */
    code, pre, [data-testid="stCode"] {{
        background-color: {bg_primary} !important;
        color: {text_primary} !important;
        border: 1px solid {border} !important;
    }}
""".format(**P)
    if not dark_mode
    else ""
)

COMMON_CSS = """
    /* ── Custom component: prediction banner ──────────────────────────────── */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(16px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes shimmer {{
        0%   {{ background-position: -200% center; }}
        100% {{ background-position: 200% center; }}
    }}

    .prediction-banner {{
        background:      {banner_bg};
        color:           {text_primary};
        border:          1px solid {banner_border};
        box-shadow:      {banner_shadow};
        padding:         2.25rem 2rem;
        border-radius:   20px;
        text-align:      center;
        margin:          1.75rem 0;
        font-size:       1.8rem;
        font-weight:     800;
        letter-spacing:  -0.02em;
        line-height:     1.3;
        animation:       fadeInUp 0.55s cubic-bezier(0.22, 1, 0.36, 1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        position:        relative;
        overflow:        hidden;
    }}

    .prediction-banner::before {{
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(
            90deg,
            transparent 0%,
            {accent_glow} 50%,
            transparent 100%
        );
        background-size: 200% auto;
        animation: shimmer 3s linear infinite;
        border-radius: inherit;
        pointer-events: none;
    }}

    /* ── VS label ────────────────────────────────────────────────────────── */
    .vs-text {{
        color:       {accent};
        text-align:  center;
        font-size:   1.35rem;
        font-weight: 900;
        margin-top:  1.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        opacity: 0.85;
    }}

    /* ── Primary button ──────────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {accent} 0%, {win_color} 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
        transition: transform 0.18s cubic-bezier(0.34, 1.56, 0.64, 1),
                    box-shadow 0.18s ease,
                    filter 0.18s ease !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        transform:  translateY(-3px) !important;
        box-shadow: 0 8px 24px {accent_glow} !important;
        filter:     brightness(1.1) !important;
    }}
    .stButton > button[kind="primary"]:active {{
        transform: translateY(-1px) !important;
    }}

    /* ── Title ───────────────────────────────────────────────────────────── */
    h1 {{
        background: linear-gradient(135deg, {accent} 0%, {text_primary} 65%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900 !important;
        letter-spacing: -0.03em !important;
    }}
""".format(**P)

st.markdown(
    f"<style>{LIGHT_OVERRIDE_CSS}{COMMON_CSS}</style>",
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏆 World Cup 2026 Predictor")
st.markdown(
    "Powered by the **Segment-Aware Hybrid Ensemble** — "
    "ELO ratings · Form · Draw-specialist · Shadow deployment."
)
st.divider()

# ── Mode tabs ─────────────────────────────────────────────────────────────────
tab_predict, tab_bracket = st.tabs(["🔮 Match Predictor", "🏆 Bracket Simulation"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Match Predictor
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    # ── Match Setup ───────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([5, 1, 5])

    with col1:
        home = st.selectbox(
            "🏠 Home Team",
            options=ALL_TEAMS,
            index=ALL_TEAMS.index("Argentina"),
            key="home_team",
        )

    with col2:
        st.markdown(
            "<div class='vs-text'>VS</div>",
            unsafe_allow_html=True,
        )

    with col3:
        away = st.selectbox(
            "✈️ Away Team",
            options=ALL_TEAMS,
            index=ALL_TEAMS.index("France"),
            key="away_team",
        )

    col_a, col_b = st.columns(2)
    with col_a:
        tournament = st.selectbox(
            "🏟️ Tournament",
            options=[
                "FIFA World Cup",
                "FIFA World Cup Qualifier",
                "UEFA Euro",
                "Copa America",
                "CONCACAF Gold Cup",
                "Africa Cup of Nations",
                "Friendly",
                "Other",
            ],
            index=0,
        )
    with col_b:
        neutral_ground = st.checkbox(
            "⚖️ Neutral Ground",
            value=True,
            help="World Cup matches are typically played on neutral ground.",
        )

    st.markdown("")

    predict_btn = st.button(
        "🔮 Predict Matchup",
        use_container_width=True,
        type="primary",
        disabled=(home == away),
    )

    if home == away:
        st.caption("⚠️ Select two different teams.")

    # ── Prediction result ──────────────────────────────────────────────────────
    if predict_btn:
        with st.spinner("Analyzing ELO, form & draw tendency…"):
            result = get_prediction(home, away, tournament, neutral_ground)

        if result:
            outcome = result["predicted_outcome"]

            outcome_display = {
                "home_win": f"🏃 **{home} wins!**",
                "away_win": f"✈️ **{away} wins!**",
                "draw": "🤝 **Draw**",
            }.get(outcome, outcome)

            predicted_score = result.get("predicted_score")
            if predicted_score:
                outcome_display += (
                    f"<br><span style='font-size: 1.1rem; font-weight: 500; "
                    f"opacity: 0.85;'>Predicted Score: {predicted_score}</span>"
                )

            st.markdown(
                f"<div class='prediction-banner'>🎯 {outcome_display}</div>",
                unsafe_allow_html=True,
            )

            probs = result.get("class_probabilities", {})
            win_home = probs.get("home_win", probs.get("Home Win", 0.0)) * 100
            draw_pct = probs.get("draw", probs.get("Draw", 0.0)) * 100
            win_away = probs.get("away_win", probs.get("Away Win", 0.0)) * 100

            m1, m2, m3 = st.columns(3)
            m1.metric(f"🏠 {home}", f"{win_home:.1f}%")
            m2.metric("🤝 Draw", f"{draw_pct:.1f}%")
            m3.metric(f"✈️ {away}", f"{win_away:.1f}%")

            # ── Advanced Analytics (collapsible) ──────────────────────────────
            with st.expander("📊 Advanced Analytics & Model Telemetry"):
                st.markdown("#### 🔎 Model Explainability")
                st.info(
                    "💡 **Why did the model predict this?**\n\n"
                    f"The **Segment-Aware Hybrid Ensemble** classified this as a "
                    f"**{'World Cup' if 'World Cup' in tournament else tournament}** fixture. "
                    "Inference was driven by:\n"
                    "- 📈 **ELO differential** between teams (recent time-decay applied)\n"
                    "- 🏃 **Form** (avg goals, conceded, win-rate over last 5)\n"
                    "- 🤝 **Draw propensity** — if probabilities fall in the uncertainty zone, "
                    "a dedicated Draw Specialist activates\n"
                    "- 🏟️ **Home advantage effect** (set to 0 for neutral ground)\n"
                    "- 💰 **Talent differential** (Transfermarkt squad values, log-scaled)"
                )

                st.markdown("#### 📡 Raw API Payload")
                meta_cols = st.columns(2)
                meta_cols[0].markdown(
                    f"**Feature source:** `{result.get('feature_source', 'N/A')}`"
                )
                meta_cols[1].markdown(
                    f"**Segment:** `{result.get('match_segment', 'N/A')}`"
                )
                meta_cols[0].markdown(
                    f"**Specialist override:** `{result.get('is_override_triggered', False)}`"
                )
                meta_cols[1].markdown(
                    f"**Model:** `{result.get('model_artifact_path', 'N/A').split(chr(92))[-1].split('/')[-1]}`"
                )

                if result.get("shadow_predicted_outcome"):
                    st.markdown("#### 🕵️ Shadow Model Comparison")
                    shadow_probs = result.get("shadow_class_probabilities", {})
                    sh_home = (
                        shadow_probs.get("home_win", shadow_probs.get("Home Win", 0.0))
                        * 100
                    )
                    sh_draw = (
                        shadow_probs.get("draw", shadow_probs.get("Draw", 0.0)) * 100
                    )
                    sh_away = (
                        shadow_probs.get("away_win", shadow_probs.get("Away Win", 0.0))
                        * 100
                    )
                    sh1, sh2, sh3 = st.columns(3)
                    sh1.metric(
                        f"🏠 {home}",
                        f"{sh_home:.1f}%",
                        delta=f"{sh_home - win_home:+.1f}%",
                    )
                    sh2.metric(
                        "🤝 Draw",
                        f"{sh_draw:.1f}%",
                        delta=f"{sh_draw - draw_pct:+.1f}%",
                    )
                    sh3.metric(
                        f"✈️ {away}",
                        f"{sh_away:.1f}%",
                        delta=f"{sh_away - win_away:+.1f}%",
                    )

                st.markdown("#### 🗄️ Full Response JSON")
                st.json(result)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bracket Simulation
# ══════════════════════════════════════════════════════════════════════════════
with tab_bracket:
    st.markdown(
        "Run **10,000 Monte Carlo simulations** of the entire knockout bracket "
        "to calculate each team's probability of advancing to every stage."
    )

    # Plausible Round of 16 bracket using real teams from 2026 fixtures
    example_bracket = [
        ["Argentina", "Canada"],
        ["England", "Uzbekistan"],
        ["France", "Ivory Coast"],
        ["Brazil", "Paraguay"],
        ["Portugal", "South Korea"],
        ["Germany", "Haiti"],
        ["Uruguay", "Bosnia and Herzegovina"],
        ["Spain", "Morocco"],
    ]

    with st.expander("📝 View / Edit Round of 16 Bracket", expanded=True):
        edited_bracket = []
        for i, match in enumerate(example_bracket):
            c1, c2 = st.columns(2)
            with c1:
                t1 = st.selectbox(
                    f"Match {i + 1} — Team 1",
                    options=ALL_TEAMS,
                    index=ALL_TEAMS.index(match[0]),
                    key=f"b_{i}_1",
                )
            with c2:
                t2 = st.selectbox(
                    f"Match {i + 1} — Team 2",
                    options=ALL_TEAMS,
                    index=ALL_TEAMS.index(match[1]),
                    key=f"b_{i}_2",
                )
            edited_bracket.append([t1, t2])

    bracket_btn = st.button(
        "🏅 Simulate Knockout Bracket",
        use_container_width=True,
        type="primary",
    )

    if bracket_btn:
        with st.spinner("Simulating the entire bracket 10,000 times…"):
            try:
                res = requests.post(
                    f"{API_URL}/simulate/bracket",
                    json={
                        "matchups": edited_bracket,
                        "n_simulations": 10000,
                        "tournament": "FIFA World Cup",
                    },
                    timeout=60,
                )
                res.raise_for_status()
                bracket_data = res.json()["probabilities"]

                import pandas as pd

                df_data = []
                for team, probs in bracket_data.items():
                    row = {"Team": team}
                    for stage, prob in probs.items():
                        row[stage.replace("_", " ")] = f"{prob * 100:.1f}%"
                    df_data.append(row)

                df = pd.DataFrame(df_data)
                df["Winner_raw"] = [bracket_data[t]["Winner"] for t in df["Team"]]
                df = df.sort_values("Winner_raw", ascending=False).drop(
                    columns=["Winner_raw"]
                )

                st.success(f"✅ Simulation complete — {len(df)} teams ranked!")
                st.dataframe(df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Failed to run bracket simulation: {e}")
