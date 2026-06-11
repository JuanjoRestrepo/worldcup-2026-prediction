"""Streamlit Frontend for World Cup 2026 Prediction Engine."""

from __future__ import annotations

import os
from typing import Any

import plotly.graph_objects as go
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

# The /predict endpoint needs no auth key — only admin routes do.
# We keep the env var for admin endpoints in case they're called later.
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

# Comprehensive team list (all major World Cup 2026 participants + more)
ALL_TEAMS = sorted(
    [
        "Argentina",
        "France",
        "Brazil",
        "England",
        "Spain",
        "Portugal",
        "Colombia",
        "Uruguay",
        "Mexico",
        "United States",
        "Germany",
        "Italy",
        "Netherlands",
        "Belgium",
        "Croatia",
        "Denmark",
        "Switzerland",
        "Morocco",
        "Senegal",
        "Nigeria",
        "Cameroon",
        "Ghana",
        "Algeria",
        "Japan",
        "South Korea",
        "Australia",
        "Iran",
        "Saudi Arabia",
        "Ecuador",
        "Chile",
        "Peru",
        "Venezuela",
        "Bolivia",
        "Paraguay",
        "Canada",
        "Costa Rica",
        "Panama",
        "Jamaica",
        "Poland",
        "Czech Republic",
        "Serbia",
        "Hungary",
        "Romania",
        "Turkey",
        "Ukraine",
        "Austria",
        "Scotland",
        "Wales",
        "Tunisia",
        "Egypt",
        "South Africa",
        "Ivory Coast",
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


# ── Theme Engine (rewrites config.toml so ALL native widgets switch correctly) ─
# Injecting CSS is unreliable for Streamlit's native elements (labels, sidebar,
# toolbar icons, checkboxes). Writing config.toml and calling st.rerun() is the
# only guaranteed way to switch the entire theme engine including the header bar.

_DARK_TOML = """\
[theme]
base="dark"
primaryColor="#0066cc"
backgroundColor="#0d1117"
secondaryBackgroundColor="#161b22"
textColor="#e6edf3"
font="sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false
"""

_LIGHT_TOML = """\
[theme]
base="light"
primaryColor="#0066cc"
backgroundColor="#f8f9fa"
secondaryBackgroundColor="#ffffff"
textColor="#212529"
font="sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false
"""

import pathlib  # noqa: E402 (needed here for config path resolution)

_CONFIG_PATH = pathlib.Path(__file__).parent.parent / ".streamlit" / "config.toml"

if "dark_mode" not in st.session_state:
    # Read the current config to initialise state correctly
    current = _CONFIG_PATH.read_text(encoding="utf-8") if _CONFIG_PATH.exists() else ""
    st.session_state["dark_mode"] = "dark" in current

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    new_dark = st.toggle("🌙 Dark Mode", value=st.session_state["dark_mode"])
    if new_dark != st.session_state["dark_mode"]:
        st.session_state["dark_mode"] = new_dark
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(
            _DARK_TOML if new_dark else _LIGHT_TOML, encoding="utf-8"
        )
        st.rerun()
    st.divider()
    st.caption("v0.1.0-alpha | Supabase Migrated")

dark_mode: bool = st.session_state["dark_mode"]

# ── CSS: only things the theme engine cannot control (custom HTML elements) ───
_DARK_BANNER_CSS = """
    .prediction-banner {
        background: rgba(0, 102, 204, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.75rem;
        font-weight: 800;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
    .vs-text { color: #0066cc !important; }
"""

_LIGHT_BANNER_CSS = """
    .prediction-banner {
        background: linear-gradient(135deg, #003d79 0%, #0066cc 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.75rem;
        font-weight: 800;
        box-shadow: 0 4px 15px rgba(0, 61, 121, 0.2);
    }
    .vs-text { color: #003d79 !important; }
"""

st.markdown(
    f"""
    <style>
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .prediction-banner {{ animation: fadeIn 0.8s ease-out; }}
    .vs-text {{
        text-align: center; font-size: 1.4rem;
        font-weight: 700; margin-top: 1.8rem;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.4);
    }}
    {"".join(c.strip() for c in (_DARK_BANNER_CSS if dark_mode else _LIGHT_BANNER_CSS).splitlines())}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🏆 World Cup 2026 Predictor")
st.markdown(
    "Powered by the **Segment-Aware Hybrid Ensemble** — "
    "ELO ratings · Form · Draw-specialist · Shadow deployment."
)
st.divider()

# ── Match Setup ───────────────────────────────────────────────────────────────
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

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Analyzing ELO, form & draw tendency…"):
        result = get_prediction(home, away, tournament, neutral_ground)

    if result:
        outcome = result["predicted_outcome"]

        # Outcome banner
        outcome_display = {
            "home_win": f"🏃 **{home} wins!**",
            "away_win": f"✈️ **{away} wins!**",
            "draw": "🤝 **Draw**",
        }.get(outcome, outcome)

        predicted_score = result.get("predicted_score")
        if predicted_score:
            outcome_display += f"<br><span style='font-size: 1.2rem; font-weight: normal;'>Exact Score: {predicted_score}</span>"

        st.markdown(
            f"<div class='prediction-banner'>🎯 {outcome_display}</div>",
            unsafe_allow_html=True,
        )

        # Raw probabilities
        probs = result.get("class_probabilities", {})
        win_home = probs.get("home_win", probs.get("Home Win", 0.0)) * 100
        draw_pct = probs.get("draw", probs.get("Draw", 0.0)) * 100
        win_away = probs.get("away_win", probs.get("Away Win", 0.0)) * 100

        # Metric cards
        m1, m2, m3 = st.columns(3)
        m1.metric(f"🏠 {home}", f"{win_home:.1f}%")
        m2.metric("🤝 Draw", f"{draw_pct:.1f}%")
        m3.metric(f"✈️ {away}", f"{win_away:.1f}%")

        # Horizontal stacked bar chart
        if win_home + draw_pct + win_away > 0:
            fig = go.Figure()
            for label, value, color in [
                (home, win_home, "#003d79"),
                ("Draw", draw_pct, "#7f8c8d"),
                (away, win_away, "#e74c3c"),
            ]:
                fig.add_trace(
                    go.Bar(
                        name=label,
                        x=[value],
                        y=[""],
                        orientation="h",
                        marker_color=color,
                        text=f"{value:.1f}%",
                        textposition="inside",
                        insidetextanchor="middle",
                        hovertemplate=f"{label}: {value:.1f}%<extra></extra>",
                        width=0.5,
                    )
                )
            fig.update_layout(
                barmode="stack",
                height=100,
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
                xaxis={"visible": False, "range": [0, 100]},
                yaxis={"visible": False},
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                bargap=0,
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        # ── Advanced Analytics (collapsible) ─────────────────────────────
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
                "- 🏟️ **Home advantage effect** (set to 0 for neutral ground)"
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
                sh_draw = shadow_probs.get("draw", shadow_probs.get("Draw", 0.0)) * 100
                sh_away = (
                    shadow_probs.get("away_win", shadow_probs.get("Away Win", 0.0))
                    * 100
                )
                sh1, sh2, sh3 = st.columns(3)
                sh1.metric(
                    f"🏠 {home}", f"{sh_home:.1f}%", delta=f"{sh_home - win_home:+.1f}%"
                )
                sh2.metric(
                    "🤝 Draw", f"{sh_draw:.1f}%", delta=f"{sh_draw - draw_pct:+.1f}%"
                )
                sh3.metric(
                    f"✈️ {away}", f"{sh_away:.1f}%", delta=f"{sh_away - win_away:+.1f}%"
                )

            st.markdown("#### 🗄️ Full Response JSON")
            st.json(result)
