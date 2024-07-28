from langchain_google_genai import ChatGoogleGenerativeAI
from data.const import GraphState
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from data.prompt_templates.advanced_template import prompt_template
from data.const import env_genai, env_openai

# from langchain_core.output_parsers import CommaSeparatedListOutputParser
from module.llm.BasicTripletChains import gpt_chain, gemini_chain


def advanced_question(chat_state: GraphState, selected_model, rewrited_question):
    output_parser = JsonOutputParser()
    # output_parser = CommaSeparatedListOutputParser()
    chat_history_str = "\n".join(
        [f"ユーザー: {q}\nAI: {a}" for q, a in chat_state["chat_history"]]
    )
    if selected_model == "Gemini_1.5_Flash":
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=env_genai,
            temperature=0,
            convert_system_message_to_human=True,
        )
    elif selected_model == "ChatGPT_3.5":
        llm = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo-0125", openai_api_key=env_openai
        )
    elif selected_model == "ChatGPT_4o_mini":
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=env_openai)

    """
    Intermediary question: {rewrited_question}
    Hint1: {hint}
    Chat history: {history}
    Final question: {question}
    Context: {context}
    Web: {web}
    """

    prompt = prompt_template
    prompt = prompt.partial(
        context=chat_state["context"],
        web=chat_state["web"],
        history=state["chat_history"],
        question=chat_state["question"],
        hint=chat_state["hint"],
    )
    input = {"rewrited_question": rewrited_question}

    chain = prompt | llm | output_parser

    advance_output_json = chain.invoke(input)
    print(advance_output_json)
    return advance_output_json


def normal_question(state: GraphState, model):
    # output_parser = CommaSeparatedListOutputParser()
    chat_history_str = "\n".join(
        [f"ユーザー: {q}\nAI: {a}" for q, a in state["chat_history"]]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたはフレンドリーなAIアシスタントです。ユーザーと話をしたりの質問に親切に答えてください",
            ),
            (
                "human",
                f"以下のチャット履歴(chat_history)を参考にしてユーザーの質問(question)に答えてください。\n\n#チャット履歴: {chat_history_str}\n\n#質問: {{question}}\n\n#Your_answer:",
            ),
        ]
    )
    prompt = prompt.partial()
    input = {"question": state["question"]}
    if model == "gemini":
        output = gemini_chain(prompt, state, input)
        print("\n\n!!!Gemini normal answer :", output)
    elif model == "gpt":
        output = gpt_chain(prompt, state, input)
        print("\n\n!!!GPT normal answer :", output)
    state["answer"] = output
    return state


if __name__ == "__main__":
    state = GraphState(
        question="日本語首都は？",
        context="",
        web="",
        answer="",
        relevance="",
        chat_history=[],
    )

    normal_question(state, "gemini")
    normal_question(state, "gpt")
    state["question"] = "fundastaの有給休暇について説明してください"
    advanced_question(state, "gemini")
    advanced_question(state, "gpt")
