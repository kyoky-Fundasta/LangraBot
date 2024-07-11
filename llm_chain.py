from sys import api_version
from const import GraphState, env_openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser


def llm_chain(state: GraphState, model):
    output_parser = CommaSeparatedListOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは親切AIアシスタントです。ユーザーの質問に対して就業規則とウェブ検索結果をもとに簡単で明瞭な答えを生成してください。質問が複数ある時には最後の質問に答えてください。答えが見つからなかった場合には丁寧に誤って'情報が見つからなかったためお答えできません'と言ってください",
            ),
            (
                "human",
                "就業規則(domument)とウェブ検索結果(Web_Search)を参考にしてユーザーの質問(question)に答えてください。答えが見つからなかった場合には丁寧に誤って'情報が見つからなかったためお答えできません'と言ってください。\n\n#就業規則: {document}\n\n#ウェブ検索結果: {web_search}\n\n#質問: {question}\n\n#Your_answer:",
            ),
        ]
    )
    prompt = prompt.partial(document=state["context"], web_search=state["web"])
    llm = ChatOpenAI(model_name=model, openai_api_key=env_openai)
    chain = prompt | llm | output_parser
    output = chain.invoke({"question": state["question"]})
    state["answer"] = output
    return state


def llm_chain_normal(state: GraphState, model):
    output_parser = CommaSeparatedListOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは親切AIアシスタントです。ユーザーと話をしたりの質問に親切に答えてください",
            ),
            (
                "human",
                "{question}\n\n#Your_answer:",
            ),
        ]
    )
    prompt = prompt.partial()
    llm = ChatOpenAI(model_name=model, openai_api_key=env_openai)
    chain = prompt | llm | output_parser
    output = chain.invoke({"question": state["question"]})
    state["answer"] = output
    return state


if __name__ == "__main__":
    state = GraphState(
        question="日本語首都は？",
        context="",
        web="",
        answer="",
        relevance="",
    )

    llm_chain(state, "gpt-3.5-turbo-0125")
    # print(state)
