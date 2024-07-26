# %%

import os
import re


# from graphviz import Digraph
# from graphviz import Graph
# import re
# from dotenv import load_dotenv
# from typing import TypedDict

# from langchain_upstage import UpstageGroundednessCheck
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# from module.vector.pinecone import retrieve_document
from module.web.tavily import search_on_web

from data.const import (
    env_openai,
    env_genai,
    index_name,
    embedding_model,
    env_tavily,
    env_pinecone,
    GraphState,
    gpt_model_name,
)
from module.llm.get_response import advanced_question, normal_question
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display

import pprint
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig
from module.llm.my_agent import ai_agent
from module.llm.relevancy_test import groundedness_check


# Question-Answer check
def relevance_check(state: GraphState) -> GraphState:
    print("\n\n$$$$$$check!!!!!!!$$$$$$$\n\n")
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

    model = ChatOpenAI(temperature=0, model=gpt_model_name, openai_api_key=env_openai)
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"question": state["question"], "answer": state["answer"]})
    print("Check response :", response)
    return GraphState(
        question=state["question"],
        context=state["context"],
        answer=state["answer"],
        relevance=response,
        chat_history=state["chat_history"],
        web=state["web"],
    )


# First Question-Answer check
def relevance_check_first(state: GraphState) -> GraphState:
    # Double check. result: Yes, No, Not_sure
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "ユーザーの質問の意図を把握してください。ユーザーは株式会社FundastAについて聞いてますか。株式会社FundastAについて聞いてる場合には(Yes)、それ以外の場合は(No)と答えてください",
            ),
            (
                "human",
                "ユーザーの質問を読んでください"
                "\n\nユーザーの質問:\n ------- \n{question}\n ------- \n"
                "\n\nある会社について聞いてるけど社名を言わなかった場合にはFundastAについて聞いて聞いてると判定して(Yes)と答えてください"
                "\n\n必ず'Yes'か'No'で答えてください"
                "\n\n説明は要りません。あなたの答えだけ返してください。",
            ),
        ]
    )

    model = ChatOpenAI(temperature=0, model=gpt_model_name, openai_api_key=env_openai)
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"question": state["question"]})

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
def chat(user_question, chat_history, model_name, who):
    workflow = StateGraph(GraphState)

    # Generate answer with LLM
    chat_state = GraphState(
        question=user_question,
        context="",
        web="",
        answer="",
        relevance="",
        chat_history=chat_history,
    )

    if who == "Guest":
        final_answer = normal_question(model_name, chat_state)
        return final_answer + "\n👦 Guest mode"

    elif who == "FundastA_社員":
        chat_state_agent = ai_agent(model_name, chat_state)
        groundedness_check_agent = groundedness_check(model_name, chat_state_agent)

        workflow.add_node("retrieve", retrieve_document)
        workflow.add_node("llm_answer", advanced_question)
        workflow.add_node("llm_answer_continue", advanced_question)
        workflow.add_node("rewrite", rewrite)
        workflow.add_node("search_on_web", search_on_web)
        workflow.add_node("relevance_check", relevance_check)
        workflow.add_node("relevance_check_continue", relevance_check)
        # Connect nodes to each other
        workflow.add_edge("retrieve", "llm_answer")
        workflow.add_edge("llm_answer", "relevance_check")
        workflow.add_edge("relevance_check", "search_on_web")
        workflow.add_edge("search_on_web", "llm_answer_continue")
        workflow.add_edge("llm_answer_continue", "relevance_check_continue")
        workflow.add_edge("rewrite", "search_on_web")

        # If statement
        workflow.add_conditional_edges(
            "relevance_check",
            is_relevant,
            {
                "Yes": END,  # Finish
                "No": "search_on_web",  # Rewrite
                "Not_sure": "search_on_web",  # Rewrite
            },
        )
        workflow.add_conditional_edges(
            "relevance_check_continue",
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
        try:
            graph = app.get_graph(xray=True)
            # Using draw_mermaid_png to render the graph
            png_bytes = graph.draw_mermaid_png()
            if png_bytes:
                # Display the graph
                display(Image(png_bytes))
            else:
                print("No PNG bytes generated")
        except Exception as e:
            print(f"An error occurred: {e}")

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
                    pprint.pprint(f"\nOutput from node '{key}':")
                    pprint.pprint("---")
                    pprint.pprint(value, indent=2, width=80, depth=None)
                    if key == "llm_answer" and "answer" in value:
                        final_answer = value["answer"]

                    if key == "llm_answer_continue" and "answer" in value:
                        final_answer = value["answer"]

                    elif key == "relevance_check_continue" and "relevance" in value:
                        relevancy = value["relevance"]

                    elif key == "relevance_check" and "relevance" in value:
                        relevancy = value["relevance"]

        except GraphRecursionError as e:
            print(f"Recursion limit reached: {e}")

        flag_2 = ""
        flag_3 = ""
        if GraphState["context"] != "":
            flag_2 = "Context available"
        if GraphState["web"] != "":
            flag_3 = "Web search available"

        print(
            "\n\n***Final Answer*** :",
            final_answer,
            "relevancy:",
            relevancy,
            flag_1,
            flag_2,
            flag_3,
        )
        return final_answer + "💛"


# Example usage
if __name__ == "__main__":
    chat_history = []  # Initialize chat history
    question_1 = "FundastAの資本金はいくらですか"
    answer_1 = chat(question_1, chat_history)


# %%
