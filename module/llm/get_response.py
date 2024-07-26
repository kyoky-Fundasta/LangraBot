from data.const import GraphState
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# from langchain_core.output_parsers import CommaSeparatedListOutputParser
from module.llm.BasicTripletChains import gpt_chain, gemini_chain


def advanced_question(state: GraphState, model):
    # output_parser = CommaSeparatedListOutputParser()
    chat_history_str = "\n".join(
        [f"ユーザー: {q}\nAI: {a}" for q, a in state["chat_history"]]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたはフレンドリーなAIアシスタントです。ユーザーの質問に対して就業規則とウェブ検索結果をもとに簡単で明瞭な答えを生成してください。質問が複数ある時には最後の質問に答えてください。答えが見つからなかった場合には丁寧に誤って'情報が見つからなかったためお答えできません'と言ってください",
            ),
            (
                "human",
                f"以下のチャット履歴(chat_history)、就業規則(document)とウェブ検索結果(Web_Search)を参考にしてユーザーの質問(question)に答えてください。答えが見つからなかった場合には丁寧に誤って'情報が見つからなかったためお答えできません'と言ってください。\n\n#チャット履歴: {chat_history_str}\n\n#就業規則: {{document}}\n\n#ウェブ検索結果: {{web_search}}\n\n#質問: {{question}}\n\n#Your_answer:",
            ),
        ]
    )
    prompt = prompt.partial(
        document=state["context"],
        web_search=state["web"],
        chat_history=state["chat_history"],
    )
    input = {"question": state["question"]}
    if model == "gemini":
        output = gemini_chain(prompt, state, input)
        print("\n\n!!!Gemini advanced answer :", output)
    elif model == "gpt":
        output = gpt_chain(prompt, state, input)
        print("\n\n!!!GPT advanced answer :", output)
    state["answer"] = output
    return state


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
