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
        st.experimental_rerun()  # Redirect to chat after successful login
    else:
        st.error("Login failure")
        st.write("Error details:", tokens)
else:
    # If not in callback phase, automatically redirect to the Cognito login page
    st.markdown(
        f"""
    <script type="text/javascript">
        window.location.href = "{login_url}";
    </script>
    """,
        unsafe_allow_html=True,
    )
    st.stop()
