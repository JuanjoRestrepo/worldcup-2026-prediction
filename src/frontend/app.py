"""Streamlit Frontend for World Cup Prediction API."""

import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Configuration
st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="🏆",
    layout="centered"
)

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "worldcup2026_super_secret_admin_key")

def get_prediction(home_team: str, away_team: str, neutral: bool = True) -> dict:
    """Fetch prediction from FastAPI backend."""
    headers = {"X-Admin-Key": ADMIN_API_KEY}
    payload = {
        "home_team": home_team,
        "away_team": away_team,
        "tournament": "FIFA World Cup",
        "neutral": neutral
    }
    
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to Prediction API. Error: {e}")
        return None

# --- UI Header ---
st.title("🏆 MLOps World Cup 2026 Engine")
st.markdown("Predict any matchup using the unified **Segment-Aware Hybrid Ensemble**.")

# --- Match Selection ---
# Some top teams to pre-populate visually
top_teams = ["Argentina", "France", "Brazil", "England", "Spain", "Portugal", "Colombia", "Uruguay", "Mexico", "United States", "Germany", "Italy"]

col1, col2, col3 = st.columns([2, 1, 2])
with col1:
    home = st.selectbox("Team 1 (Home/Nominal)", options=top_teams, index=0)
with col2:
    st.markdown("<h3 style='text-align: center; margin-top:25px;'>VS</h3>", unsafe_allow_html=True)
with col3:
    away = st.selectbox("Team 2 (Away/Nominal)", options=top_teams, index=1)

neutral_ground = st.checkbox("Neutral Ground (World Cup Format)", value=True)

if st.button("🔮 Predict Matchup", use_container_width=True, type="primary"):
    if home == away:
        st.warning("Please select two different teams.")
    else:
        with st.spinner("Analyzing historical ELO, form, and draw tendencies..."):
            result = get_prediction(home, away, neutral_ground)
            
        if result:
            st.success("Prediction Generated Successfully!")
            st.markdown(f"### 🎯 The Model Predicts: **{result['predicted_outcome']}**")
            
            probs = result['class_probabilities']
            win_home = probs.get("Home Win", 0.0) * 100
            draw = probs.get("Draw", 0.0) * 100
            win_away = probs.get("Away Win", 0.0) * 100
            
            # --- Visual Probability Bar ---
            fig_bar = go.Figure(go.Bar(
                x=[win_home, draw, win_away],
                y=["Probability"],
                orientation='h',
                marker=dict(color=['#1f77b4', '#7f7f7f', '#ff7f0e']),
                text=[f"{win_home:.1f}% ({home})", f"{draw:.1f}% (Draw)", f"{win_away:.1f}% ({away})"],
                textposition='auto',
                width=0.4
            ))
            fig_bar.update_layout(
                barmode='stack',
                height=200,
                xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 100]),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                margin=dict(l=0, r=0, t=0, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

            # --- Advanced Analytics Frame ---
            with st.expander("📊 Explanatory Analytics & Telemetry (For Engineers)"):
                st.markdown("**(Feature Importance & Model Telemetry)**")
                st.write(f"Inference logged to Database tracking ID: `{result.get('telemetry_id', 'Not Provided')}`")
                
                # We show the raw JSON so users and engineers can inspect the microservice payload
                st.json(result)
                
                # Mockup/Explanation of SHAP Logic:
                st.info("💡 **Why did the model pick this?**\n"
                        "Our *Segment-Aware Hybrid Model* isolates World Cup matches. It strictly evaluated the "
                        "Elo Difference, Form, and Recent Draw tendency of these two exact configurations. "
                        "If the probabilities are extremely close (Uncertainty zone), a Secondary Draw-Specialist was engaged to prevent under-predicting draws.")
