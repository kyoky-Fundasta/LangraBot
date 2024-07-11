# %%
import os

# import re

from dotenv import load_dotenv
from typing import TypedDict

# from langchain_upstage import UpstageGroundednessCheck
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from rag.utils import format_docs, format_searched_docs
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from const import env_openai, index_name, embedding_model, env_tavily
from llm_chain import llm_chain
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# from IPython.display import Image, display
# from graphviz import Digraph
import pprint
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig

from test_sample import tavily_result1


# Define GraphState including chat_history
class GraphState(TypedDict):
    question: str
    context: str
    web: str
    answer: str
    relevance: str
    chat_history: list  # Added chat_history


embeddings = OpenAIEmbeddings(openai_api_key=env_openai, model=embedding_model)
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)
retriever = docsearch.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "fetch_k": 6}
)
os.environ["TAVILY_API_KEY"] = env_tavily
model_name = "gpt-3.5-turbo-0125"


#   RAG document retrieval
def retrieve_document(state: GraphState) -> GraphState:
    # Retrieves related reference from VectorDB
    retrieved_docs = retriever.invoke(state["question"])
    # Reshape the data
    retrieved_docs = format_docs(retrieved_docs)
    # Preserve it in a GraphState
    return GraphState(
        question=state["question"],
        context=retrieved_docs,
        chat_history=state["chat_history"],
        web=state["web"],
        answer=state["answer"],
        relevance=state["relevance"],
    )


# Generate an answer using LLM
def llm_answer(state: GraphState) -> GraphState:
    # Generate AI answer
    response_state = llm_chain(state, model_name)
    return GraphState(
        question=state["question"],
        answer=response_state["answer"],
        context=state["context"],
        chat_history=state["chat_history"],
        web=state["web"],
        relevance=state["relevance"],
    )


# Rewrite the user question
def rewrite(state: GraphState) -> GraphState:
    question = state["question"]
    answer = state["answer"]
    context = state["context"]
    web = state["web"]
    chat_history = state["chat_history"]
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
                "\n\nHere is the chat history:\n ------- \n{chat_history}\n ------- \n"
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
        {
            "chat_history": chat_history,
            "question": question,
            "answer": answer,
            "context": context,
        }
    )
    response = response + state["question"]
    return GraphState(
        question=response,
        context=state["context"],
        chat_history=chat_history,
        web=state["web"],
        answer=state["answer"],
        relevance=state["relevance"],
    )


# Web search API
def search_on_web(state: GraphState) -> GraphState:
    # Tavily web search
    # search = TavilySearchAPIWrapper()
    # search_tool = TavilySearchResults(max_results=5, api_wrapper=search)
    # search_result = search_tool.invoke({"query": state["question"]})
    # Test data for saving tavily search api
    search_result = tavily_result1
    # print("##Tavily:", search_result)
    # Reshape the search_result
    search_result = format_searched_docs(search_result)
    # Preserve it in the state.
    # print("\n\n##Tavily2:", search_result)
    return GraphState(
        question=state["question"],
        context=state["context"],
        web=search_result,
        chat_history=state["chat_history"],
        answer=state["answer"],
        relevance=state["relevance"],
    )


# Question-Answer check
def relevance_check(state: GraphState) -> GraphState:
    # Double check. result: Yes, No, Not_sure
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたはAIアシストががユーザーの質問にちゃんと答えることが出来たかどうかを判定をするジャッジです。質問が複数ある時には最後の質問とAIアシストの回答で判定してください。ユーザーの最後の質問に対してAIアシストの回答がしっかりと関連性をもって正しく答えられていれば(Yes)、AIアシストの回答にユーザーの質問への答えが入ってなかったら(No)、判断に困難な場合は(Not_sure)と答えてください。質問が複数ある時には最後の質問とAIアシストの回答で判定してください",
            ),
            (
                "human",
                "まずユーザーの質問とAIアシストの回答を読んでください"
                "\n\nユーザーの質問:\n ------- \n{question}\n ------- \n"
                "\n\nAIアシストの回答:\n ------- \n{answer}\n ------- \n"
                "\n\nAIアシストはユーザーの質問に正しく答えることが出来ましたか。判定をしてください"
                "\n\n質問が複数ある時には最後の質問とAIアシストの回答で判定してください。"
                "\n\nあなたは'Yes'か'No'か'Not_sure'のどれかで答えなければなりません"
                "\n\n説明は要りません。あなたの答えだけ返してください。",
            ),
        ]
    )

    model = ChatOpenAI(temperature=0, model=model_name, openai_api_key=env_openai)
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"question": state["question"], "answer": state["answer"]})
    print(response)
    return GraphState(
        question=state["question"],
        context=state["context"],
        answer=state["answer"],
        relevance=response,
        chat_history=state["chat_history"],
        web=state["web"],
    )


def is_relevant(state: GraphState) -> GraphState:
    return state["relevance"]


# if __name__ == "__main__":
def chat(user_question, chat_history):
    workflow = StateGraph(GraphState)
    # Node definition
    workflow.add_node("retrieve", retrieve_document)
    workflow.add_node("llm_answer", llm_answer)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("search_on_web", search_on_web)
    workflow.add_node("relevance_check", relevance_check)
    # Connect nodes to each other
    workflow.add_edge("retrieve", "llm_answer")
    workflow.add_edge("llm_answer", "relevance_check")
    workflow.add_edge("rewrite", "search_on_web")
    workflow.add_edge("search_on_web", "llm_answer")
    # If statement
    workflow.add_conditional_edges(
        "relevance_check",
        is_relevant,
        {
            "Yes": END,  # Finish
            "No": "rewrite",  # Rewrite
            "Not_sure": "rewrite",  # Rewrite
        },
    )

    workflow.set_entry_point("retrieve")
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    #   Draw a diagram describing reasoning flow
    # try:
    #     graph = app.get_graph(xray=True)
    #     # Using draw_mermaid_png to render the graph
    #     png_bytes = graph.draw_mermaid_png()
    #     # Display the graph
    #     display(Image(png_bytes))
    # except Exception as e:
    #     print(f"An error occurred: {e}")

    config = RunnableConfig(
        recursion_limit=12, configurable={"thread_id": "CORRECTIVE-SEARCH-RAG"}
    )

    inputs = GraphState(
        question=user_question,
        chat_history=chat_history,
        context="",
        web="",
        answer="",
        relevance="",
    )
    final_answer = None
    relevancy = None
    try:
        for output in app.stream(inputs, config=config):
            for key, value in output.items():
                pprint.pprint(f"Output from node '{key}':")
                pprint.pprint("---")
                pprint.pprint(value, indent=2, width=80, depth=None)
                if key == "llm_answer" and "answer" in value:
                    final_answer = value["answer"]
                elif key == "relevance_check" and "relevance" in value:
                    relevancy = value["relevance"]
    except GraphRecursionError as e:
        print(f"Recursion limit reached: {e}")

    # print("\n\n***Final Answer*** :", final_answer, relevancy)
    return final_answer[0]


# %%

# Example usage
if __name__ == "__main__":
    chat_history = []  # Initialize chat history
    question_1 = "日本の首都はどこ"
    answer_1 = chat(question_1, chat_history)
    chat_history.append((question_1, answer_1))
    print(answer_1)

    question_2 = "そうするとフランスは。"
    answer_2 = chat(question_2, chat_history)
    chat_history.append((question_2, answer_2))
    print(answer_2)
