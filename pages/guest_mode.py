import streamlit as st
from app import chat


who = "Guest"

st.title("💬 FundastA ChatBot")
st.caption("🤖 Geminiチャットボットを無料で試せます。")


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
            model_name="Gemini_1.5_Flash",
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
