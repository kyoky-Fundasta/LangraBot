from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from data.const import GraphState
from langchain_core.prompts import ChatPromptTemplate
from data.prompt_templates.rewriter import prompt_template
from langchain_core.output_parsers import StrOutputParser
from data.const import env_genai, env_openai


# Rewrite the user question
def rewrite_question(chat_state: GraphState, selected_model):
    chat_history_str = "\n".join(chat_state["chat_history"])
    question = chat_state["question"]
    context = chat_state["context"]
    web = chat_state["web"]
    answer = chat_state["answer"]

    prompt = prompt_template

    if selected_model == "Gemini_1.5_Flash":
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=env_genai,
            temperature=0,
            convert_system_message_to_human=True,
        )
        print("Gemini selected")
    elif selected_model == "ChatGPT_3.5":
        llm = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo-0125", openai_api_key=env_openai
        )
        print("GPT selected")

    elif selected_model == "ChatGPT_4o_mini":
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=env_openai)
        print("GPT mini selected")

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(
        {
            "chat_history": chat_history_str,
            "question": question,
            "answer": answer,
            "context": context,
            "web": web,
        }
    )
    print("\n------updated question :", response)
    return response
