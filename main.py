import streamlit as st
from app import chat
from data.const import GraphState
import requests
from urllib.parse import urlparse, parse_qs
from data.const import client_id

cognito_domain = "fundasta-ai-assistant"
client_id = client_id
region = "ap-northeast-1"
redirect_uri = "https://fundasta-aibot.streamlit.app/"

login_url = f"https://{cognito_domain}.auth.{region}.amazoncognito.com/login?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}"
st.write("Attempting to connect to:", login_url)
query_params = st.experimental_get_query_params()
st.write(query_params)
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

    else:
        st.error("Login failure")
else:
    st.markdown(
        f'<meta http-equiv="refresh" content="0;url={login_url}">',
        unsafe_allow_html=True,
    )
    st.stop()


# WebUI (Streamlit)
st.set_page_config(layout="wide")
st.title("💬 FundastA サポートデスク")
st.caption("🤖 私は株式会社FundastAのAIアシストです。")


menu_options = ["ChatGPT_4o_mini", "ChatGPT_3.5", "Gemini_1.5_Flash", "Claude"]
login_options = ["FundastA_社員", "Guest"]
model = st.selectbox("LLM models", menu_options)
who = st.selectbox("Log-in options", login_options)

st.sidebar.title("MENU")
ai_bot = st.sidebar.radio("MENU", ["チャットで質問", "メールで問い合わせ", "資料検索"])

if ai_bot == "チャットで質問":

    if "message" not in st.session_state:
        st.session_state["message"] = []

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    with st.container():
        input = st.chat_input("👤質問を入力してください")

    if input:
        with st.spinner("🤖 考え中......"):
            ai_answer = chat(
                user_question=input,
                chat_history=st.session_state["chat_history"],
                model_name=model,
                who=who,
            )
            print(
                "\n\n------------Main Program From Here----------------\n\nai_answer :",
                type(ai_answer),
                ai_answer,
            )
            feedback = None
            source = None
            if ai_answer is not None:
                if who == "Guest":
                    st.session_state["message"].append(
                        {
                            "role": "assistant",
                            "content": ai_answer["answer"] + "  [👦 Guest mode]",
                        }
                    )

                elif who == "FundastA_社員":

                    if ai_answer["relevance"] == "grounded":
                        feedback = "判定：🌞　　feedback : " + ai_answer["reasoning"]
                        source = "source : " + ai_answer["source"]
                        st.session_state["message"].append(
                            {
                                "role": "assistant",
                                "content": ai_answer["answer"] + "  [🏢 社員 mode]",
                            }
                        )
                    elif ai_answer["relevance"] == None:
                        st.session_state["message"].append(
                            {
                                "role": "assistant",
                                "content": ai_answer["answer"] + "  [🏢 社員 mode]",
                            }
                        )
                    elif ai_answer["relevance"] != "grounded":
                        feedback = "\n判定：☔　　feedback : " + ai_answer["reasoning"]
                        source = "source : " + ai_answer["source"]

                        st.session_state["message"].append(
                            {
                                "role": "assistant",
                                "content": "AI：次の答えは間違ってる可能性があります。再度確認することをお勧めします。\n"
                                + ai_answer["answer"]
                                + "  [🏢 社員 mode]",
                            }
                        )

                st.session_state["message"].append(
                    {
                        "role": "user",
                        "content": input,
                    }
                )
                st.session_state["chat_history"].append((input, ai_answer["answer"]))

                if st.session_state["message"]:
                    f = 0
                    print("\n---------feedback :", feedback, source)
                    for message in st.session_state["message"][::-1]:
                        f += 1
                        print("\n---------counter :", f)
                        if f == 3 and feedback != None:
                            st.write(feedback)
                            if source != None:
                                st.write(source)
                        with st.chat_message(message["role"]):
                            st.write(message["content"])
            else:
                st.error("AI answer was None. Please check the chat function.")
                st.write(ai_answer)
elif ai_bot == "メールで問い合わせ":
    pass

elif ai_bot == "資料検索":
    pass
