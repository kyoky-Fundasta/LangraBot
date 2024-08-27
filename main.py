import streamlit as st
import requests
from urllib.parse import urlparse, parse_qs
from data.const import client_id


# Set up AWS Cognito configuration
st.set_page_config(
    page_title="FundastA AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Set up AWS Cognito configuration
cognito_domain = "fundasta-ai-assistant"
client_id = st.secrets["client_id"]
region = "ap-northeast-1"
redirect_uri = "https://fundasta-aibot.streamlit.app"

login_url = f"https://{cognito_domain}.auth.{region}.amazoncognito.com/login?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}"

st.title("FundastA AI Assistant")
col1, col2 = st.columns(2)

with col1:
    if st.button("ゲストモード"):
        st.query_params["mode"] = "guest_mode"
        st.rerun()

with col2:
    if st.button("社員モード"):
        js_code = f"""
        <script>
        window.open("{login_url}", "_self");
        </script>
        """
        st.components.v1.html(js_code, height=0, width=0)

# Check if we're in the callback phase
query_params = st.query_params

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
        st.query_params["mode"] = "employ_mode"
        st.rerun()
    else:
        st.error("Login failure")
        st.write("Error details:", tokens)
