"""Streamlit UI. Collects cc_num/amt/category, calls the API, shows the
decision. Timestamp is NOT collected — the server assigns it at ingest.
"""
import os

import requests
import streamlit as st

from src.config import get_settings

API_URL = os.getenv("FRAUD_API_URL", "http://localhost:8000")
SETTINGS = get_settings()

# Must match the Sparkov training vocabulary for an authentic demo.
CATEGORIES = [
    "gas_transport", "grocery_pos", "shopping_net", "entertainment",
    "food_dining", "health_fitness", "misc_net", "travel",
]

# Centered, larger, colourful typography.
_STYLE = """
<style>
  .stApp { background: linear-gradient(180deg, #f7f9fc 0%, #eef2f8 100%); }
  .block-container { max-width: 620px; padding-top: 2.5rem; }
  h1 { font-size: 2.6rem !important; text-align: center; color: #1a2b4a;
       font-weight: 800; margin-bottom: 0.3rem; }
  .subtitle { font-size: 1.35rem; text-align: center; color: #4a5a72;
              margin-bottom: 2rem; line-height: 1.4; }
  label, .stTextInput input, .stNumberInput input, .stSelectbox div {
      font-size: 1.15rem !important; }
  .stButton button { font-size: 1.2rem !important; font-weight: 700;
      background: #1a2b4a; color: #fff; border: none; padding: 0.6rem; }
  .stButton button:hover { background: #26406e; color: #fff; }
  .result-card { border-radius: 12px; padding: 1.4rem 1.6rem; margin-top: 1.5rem;
      text-align: center; box-shadow: 0 4px 14px rgba(0,0,0,0.08); }
  .decision { font-size: 2.4rem; font-weight: 800; margin-bottom: 0.6rem; }
  .score { font-size: 1.3rem; color: #333; font-weight: 600; }
  .counts { font-size: 1.1rem; color: #555; margin-top: 0.5rem; }
</style>
"""


def decide(score: float) -> tuple[str, str, str]:
    """Map a score to a decision, text colour, and soft background colour."""
    if score >= SETTINGS.decline_threshold:
        return "DECLINE", "#d62728", "#fdecec"
    if score >= SETTINGS.review_threshold:
        return "STEP-UP", "#e67e00", "#fff5e8"
    return "APPROVE", "#1f9d55", "#eafaf1"


def valid_cc(cc: str) -> bool:
    """Client-side gate: digits only, plausible length (server re-validates)."""
    return cc.isdigit() and 12 <= len(cc) <= 19


def main() -> None:
    """Render the centered form and the latest decision."""
    st.set_page_config(page_title="Fraud Detection", layout="centered")
    st.markdown(_STYLE, unsafe_allow_html=True)

    if "submitting" not in st.session_state:
        st.session_state.submitting = False

    st.title("Fraud Detection Demo")
    st.markdown(
        "<div class='subtitle'>Repeat the same card and watch the "
        "velocity counts from DynamoDB and the fraud score climb.</div>",
        unsafe_allow_html=True,
    )

    cc_num = st.text_input("Card number", value="000123456789")
    amt = st.number_input("Amount", min_value=0.01, value=500.0, step=1.0)
    category = st.selectbox("Category", CATEGORIES)

    cc_ok = valid_cc(cc_num)
    if cc_num and not cc_ok:
        st.error("Card number must be 12–19 digits.")

    clicked = st.button(
        "Submit transaction",
        disabled=st.session_state.submitting or not cc_ok,
        use_container_width=True,
    )

    if clicked:
        st.session_state.submitting = True  # lock the button
        try:
            with st.spinner("Scoring…"):
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"cc_num": cc_num, "amt": amt, "category": category},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                score = data["fraud_score"]
                cnt_1h = data["cc_cnt_1h"]
                cnt_24h = data["cc_cnt_24h"]

            decision, colour, bg = decide(score)

            st.markdown(
                f"<div class='result-card' style='background:{bg}'>"
                f"<div class='decision' style='color:{colour}'>{decision}</div>"
                f"<div class='score'>Fraud score: {score:.3f}</div>"
                f"<div class='counts'>Txns last 1h: {int(cnt_1h)} &nbsp;·&nbsp; "
                f"last 24h: {int(cnt_24h)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        except requests.RequestException as e:
            st.error(f"API error: {e}")
        finally:
            st.session_state.submitting = False  # unlock regardless of outcome


if __name__ == "__main__":
    main()
