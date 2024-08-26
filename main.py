import streamlit as st
import requests
from urllib.parse import urlparse, parse_qs
from data.const import client_id


# Set up AWS Cognito configuration
cognito_domain = "fundasta-ai-assistant"
client_id = st.secrets["client_id"]
region = "ap-northeast-1"
redirect_uri = (
    "https://fundasta-aibot.streamlit.app/UI"  # Redirect to chat.py after login
)

login_url = f"https://{cognito_domain}.auth.{region}.amazoncognito.com/login?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}"


st.title("FundastA AI Assistant")
col1, col2 = st.columns(2)

with col1:
    if st.button("ゲストモード"):
        st.markdown(
            f"""
            <meta http-equiv="refresh" content="0; url='/UI'">
            """,
            unsafe_allow_html=True,
        )

with col2:
    if st.button("社員モード"):
        st.markdown(
            f"""
            <meta http-equiv="refresh" content="0; url='{login_url}'">
            """,
            unsafe_allow_html=True,
        )

# Check if we're in the callback phase
query_params = st.experimental_get_query_params()

if "code" in query_params:
    auth_code = query_params["code"][0]
    st.write("Authorization code received:", auth_code)
    token_url = f"https://{cognito_domain}.auth.{region}.amazoncognito.com/oauth2/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": auth_code,
        "redirect_uri": redirect_uri,
    }

    response = requests.post(token_url, data=data)
    tokens = response.json()

    if response.status_code == 200:
        st.success("Login successful")
        st.session_state["tokens"] = tokens
        st.markdown(
            f"""
            <meta http-equiv="refresh" content="0; url='/UI'">
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("Login failure")
        st.write("Error details:", tokens)
