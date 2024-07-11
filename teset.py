# %%
import os
import re
from dotenv import load_dotenv
from typing import TypedDict
from langchain_upstage import UpstageGroundednessCheck
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from rag.utils import format_docs, format_searched_docs
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from const import env_openai, index_name, embedding_model, GraphState, env_tavily
from llm_chain import llm_chain

from test_sample import tavily_result1

embeddings = OpenAIEmbeddings(openai_api_key=env_openai, model=embedding_model)
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)
retriever = docsearch.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "fetch_k": 6}
)
os.environ["TAVILY_API_KEY"] = env_tavily
# pdf_chain = pdf.chain


#   RAG document retrieval
def retrieve_document(state: GraphState) -> GraphState:

    # Retrieves related refence from VectorDB
    retrieved_docs = retriever.invoke(state["question"])
    # Reshape the data
    retrieved_docs = format_docs(retrieved_docs)
    # Preserve it in a GraphState
    return GraphState(
        question=state["question"],
        context=retrieved_docs,
        web=state["web"],
        answer=state["answer"],
        relevance=state["relevance"],
    )


# Generate an answer using LLM
def llm_answer(state: GraphState, model_name) -> GraphState:

    # 체인을 호출하여 답변을 생성합니다. response : GraphState
    response = llm_chain(state, model_name)

    return response


def rewrite(state, model_name):
    question = state["question"]
    answer = state["answer"]
    context = state["context"]
    web = state["web"]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional prompt rewriter. Your task is to generate the question in order to get additional information that is now shown in the context."
                "Your generated question will be searched on the web to find relevant information.",
            ),
            (
                "human",
                "Rewrite the question to get additional information to get the answer."
                "\n\nHere is the initial question:\n ------- \n{question}\n ------- \n"
                "\n\nHere is the initial context:\n ------- \n{context}\n ------- \n"
                "\n\nHere is the initial answer to the question:\n ------- \n{answer}\n ------- \n"
                "\n\nFormulate an improved question in Japanese:",
            ),
        ]
    )

    # Question rewriting model
    model = ChatOpenAI(temperature=0, model=model_name, openai_api_key=env_openai)

    chain = prompt | model | StrOutputParser()
    response = chain.invoke(
        {"question": question, "answer": answer, "context": context}
    )
    state["question"] = response + state["question"]
    return state


def search_on_web(state: GraphState) -> GraphState:
    # Tavily web search
    # search = TavilySearchAPIWrapper()
    # search_tool = TavilySearchResults(max_results=5, api_wrapper=search)
    # search_result = search_tool.invoke({"query": state["question"]})

    # Test data for savinf tavily search api
    search_result = tavily_result1

    # print("##Tavily:", search_result)
    # Reshape the search_result
    search_result = format_searched_docs(search_result)
    # Preserve it in the state.
    # print("\n\n##Tavily2:", search_result)
    state["web"] = search_result
    return state


# %%


def relevance_check(state: GraphState, model_name) -> GraphState:

    # Doubole check. result: Yes, No, notSure
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたはAIアシストががユーザーの質問にちゃんと答えることが出来たかどうかを判定をするジャッジです。質問が複数ある時には最後の質問とAIアシストの回答で判定してください。ユーザーの最後の質問に対してAIアシストの回答がしっかりと関連性をもって正しく答えられていれば(Yes)、AIアシストの回答にユーザーの質問への答えが入ってなかったら(No)、判断に困難な場合は(Not sure)と答えてください。質問が複数ある時には最後の質問とAIアシストの回答で判定してください",
            ),
            (
                "human",
                "まずユーザーの質問とAIアシストの回答を読んでください"
                "\n\nユーザーの質問:\n ------- \n{question}\n ------- \n"
                "\n\nAIアシストの回答:\n ------- \n{answer}\n ------- \n"
                "\n\nAIアシストはユーザーの質問に正しく答えることが出来ましたか。判定をしてください"
                "\n\n質問が複数ある時には最後の質問とAIアシストの回答で判定してください。"
                "\n\nあなたは'Yes'か'No'か'Not sure'のどれかで答えなければなりません"
                "\n\n説明は要りません。あなたの答えだけ返してください。",
            ),
        ]
    )

    # Question rewriting model
    model = ChatOpenAI(temperature=0, model=model_name, openai_api_key=env_openai)

    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"question": state["question"], "answer": state["answer"]})
    state["relevance"] = response
    return state


# %%
if __name__ == "__main__":
    _ = GraphState(
        question="FundastAの設立日はいつですか。",
        context="",
        web="",
        answer="",
        relevance="",
    )

    print("\n\n@1 :", _)
    gs2 = retrieve_document(_)
    print("\n\n@2 :", gs2)
    model_name = "gpt-3.5-turbo-0125"
    gs3 = llm_answer(gs2, model_name)
    print("\n\n@3 :", gs3)
    gs3_1 = relevance_check(gs3, model_name)
    print("\n\n@3_1 :", gs3_1)
    gs3_2 = rewrite(gs3_1, model_name)
    print("\n\n@3_2 :", gs3_2)
    gs4 = search_on_web(gs3_2)
    print("\n\n@4 :", gs4)
    gs5 = llm_answer(gs4, model_name)
    print("\n\n@5 :", gs5)
    gs6 = relevance_check(gs5, model_name)
    print("\n\n@6 :", gs6)


# %%
