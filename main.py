import streamlit as st
from app import chat
from data.const import GraphState


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

            if who == "Guest":
                formatted_answer = ai_answer["answer"] + "  👦 Guest mode"
                feedback = None
            elif who == "FundastA_社員":
                last_answer = ai_answer["answer"]
                if ai_answer["response_type"] == 1:
                    feedback = (
                        "判定：🌞　　feedback : "
                        + ai_answer["reasoning"]
                        + "source :"
                        + ai_answer["source"]
                    )
                elif ai_answer["response_type"] == 0:
                    feedback = None
                elif ai_answer["response_type"] == -1:
                    feedback = (
                        "関連情報が不足したため、正確な回答を作成することが来ませんでした。\n判定：☔　　feedback : "
                        + ai_answer["reasoning"]
                    )
                formatted_answer = last_answer + "  🏢 社員 mode"
        st.session_state["message"].append(
            {
                "role": "assistant",
                "content": ai_answer["answer"],
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
            for message in st.session_state["message"][::-1]:
                f += 1
                with st.chat_message(message["role"]):
                    print(f)
                    st.write(message["content"])
            st.write(feedback)

elif ai_bot == "メールで問い合わせ":
    pass

elif ai_bot == "資料検索":
    pass
