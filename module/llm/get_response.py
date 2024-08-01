from data.const import GraphState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from data.prompt_templates.advanced_question_template import prompt_template
from data.const import gemini_model_name, gpt_mini_model_name, llm_switch
from langchain_core.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from module.llm.BasicTripletChains import gpt_chain, gemini_chain


def advanced_question(chat_state: GraphState) -> GraphState:
    selected_model = chat_state["selected_model"]
    output_parser = StrOutputParser()
    # output_parser = CommaSeparatedListOutputParser()
    chat_history_str = "\n".join(
        [f"ユーザー: {q}\nAI: {a}" for q, a in chat_state["chat_history"]]
    )

    llm = llm_switch(selected_model)

    prompt = PromptTemplate.from_template(prompt_template)
    prompt = prompt.partial(
        context=chat_state["context"],
        web=chat_state["web"],
        chat_history=chat_history_str,
        hint=chat_state["hint"],
    )
    input = {"question": chat_state["question"]}

    chain = prompt | llm | output_parser

    advance_output_str = chain.invoke(input)
    print(type(advance_output_str), advance_output_str)
    chat_state["answer"] = advance_output_str
    return chat_state


def normal_question(state: GraphState):
    model = state["selected_model"]
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
    if model == "Gemini_1.5_Flash":
        output = gemini_chain(prompt, state, input)
        print("\n\n!!!Gemini normal answer :", type(output), output)
    elif model == "ChatGPT_4o_mini" or "ChatGPT_3.5":
        output = gpt_chain(prompt, state, input)
        print("\n\n!!!GPT normal answer :", type(output), output)
    state["answer"] = output
    return state


if __name__ == "__main__":

    state = GraphState(
        selected_model="",
        question="日本語首都は？",
        context="",
        web="",
        answer="",
        relevance="",
        chat_history=[],
        hint="",
    )
    state["selected_model"] = gemini_model_name
    _ = normal_question(state)
    state["selected_model"] = gpt_mini_model_name
    _ = normal_question(state)
    state["question"] = "fundastaの有給休暇について説明してください"
    _ = advanced_question(state)
    state["selected_model"] = gemini_model_name
    _ = advanced_question(state)
