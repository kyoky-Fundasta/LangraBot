import streamlit as st
from app import chat, pinecone


# WebUI (Streamlit)
st.set_page_config(layout="wide")
st.title("💬 FundastA サポートデスク")
st.caption("🤖 私は株式会社FundastAのAIアシストです。")
# st.title("test pinecone")
# if st.button("test pinecone"):
#     result = pinecone()
#     st.write(result)


menu_options = ["ChatGPT 3.5", "ChatGPT 4o", "Gemini", "Claude"]
ai_bot = st.selectbox("LLM models", menu_options)

st.sidebar.title("MENU")
ai_bot = st.sidebar.radio("", ["チャットで質問", "メールで問い合わせ", "資料検索"])

if ai_bot == "チャットで質問":

    if "message" not in st.session_state:
        st.session_state["message"] = []

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    with st.container():
        prompt = st.chat_input("👤質問を入力してください")

    if prompt:

        with st.spinner("🤖 考え中......"):
            ai_answer = chat(
                user_question=prompt, chat_history=st.session_state["chat_history"]
            )
            # # pages = [
            # #     str(int(doc.metadata.get("page-number")))
            # #     for doc in ai_answer["source_documents"]
            # # ]
            # str = ", ".join(pages)
            # formatted_answer = ai_answer["answer"]
            # formatted_answer = f'{ai_answer["answer"]}\n\n参照したページ：{str}'
            # st.write(formatted_answer)
        st.session_state["message"].append(
            {
                "role": "assistant",
                "content": ai_answer,
            }
        )
        st.session_state["message"].append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        st.session_state["chat_history"].append((prompt, ai_answer))

        if st.session_state["message"]:

            for message in st.session_state["message"][::-1]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

elif ai_bot == "メールで問い合わせ":
    pass

elif ai_bot == "資料検索":
    pass
