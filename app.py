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
from module.llm.advanced_agent import ai_advanced_agent

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
from module.llm.relevancy_test import groundedness_check, is_grounded
from module.llm.editor import rewrite_question


def summarize_final_answer(chat_state):
    final_response = {
        "question": "",
        "answer": "",
        "groundedness": "",
        "reasoning": "",
        "source": "",
    }
    if chat_state["result"] == "grounded":
        final_response["question"] = chat_state["question"]
        final_response["answer"] = chat_state["answer"]
        final_response["groundedness"] = chat_state["result"]
        final_response["reasoning"] = chat_state["reasoning"]
        final_response["source"] = chat_state["source"]
    else:
        final_response["question"] = chat_state["question"]
        final_response["answer"] = (
            "申し訳ありません。詳しい情報が見つかりませんでした。担当部署にお問い合わせください。"
        )
        final_response["groundedness"] = chat_state["result"]
        final_response["reasoning"] = chat_state["reasoning"]
        final_response["source"] = chat_state["source"]
    return final_response


def merge_states(chat_state: GraphState, agent_state: GraphState) -> GraphState:
    hint = agent_state["question"] + agent_state["answer"]
    merged_context = chat_state["context"] + agent_state["context"]
    merged_web = chat_state["web"] + agent_state["web"]
    merged_state = GraphState(
        question=chat_state["question"],
        hint=hint,
        context=merged_context,
        web=merged_web,
        answer="",
        chat_history=chat_state["chat_history"],
        relevance=chat_state["relevance"],
    )
    return merged_state


def invoke_chain(app, chat_state, selected_model):

    invoke_config = RunnableConfig(
        recursion_limit=3,
        configurable={
            "thread_id": "CORRECTIVE-SEARCH-RAG",
        },
    )

    rewrotten_state = app.invoke(
        chat_state,
        {"selected_model": selected_model},  # Removed "config" key from here
        config=invoke_config,  # Ensure config is passed correctly
    )
    agent_state = app.invoke(
        rewrotten_state,
        {"selected_model": selected_model},  # Removed "config" key from here
        config=invoke_config,  # Ensure config is passed correctly
    )
    merged_state = app.invoke(
        {"agent_state": agent_state},  # Removed "config" key from here
        config=invoke_config,
    )

    advanced_response = app.invoke(
        merged_state,
        {"selected_model": selected_model},  # Removed "config" key from here
        config=invoke_config,  # Ensure config is passed correctly
    )

    final = app.invoke(
        advanced_response,
        {"selected_model": selected_model},  # Removed "config" key from here
        config=invoke_config,  # Ensure config is passed correctly
    )
    return final


# if __name__ == "__main__":
def chat(user_question, chat_history, model_name, who):

    # Generate answer with LLMphState
    chat_state = GraphState(
        selected_model=model_name,
        question=user_question,
        context="",
        web="",
        answer="",
        relevance="",
        chat_history=chat_history,
        hint="",
        rewrotten_question="",
        rewrotten_question_answer="",
        reasoning="",
        source="",
    )

    if who == "Guest":
        final_answer = normal_question(chat_state)
        return final_answer + "\n👦 Guest mode"

    elif who == "FundastA_社員":
        chat_state_agent = ai_agent(chat_state)
        chat_state = groundedness_check(chat_state_agent)

        if chat_state["result"] == "grounded":
            _ = summarize_final_answer(chat_state)
            print("\n\n----------------Answering routine 1---------------------\n\n", _)
            return _

        else:
            workflow = StateGraph(GraphState)

            workflow.add_node("rewrite_question", rewrite_question)
            workflow.add_node("ai_advanced_agent", ai_advanced_agent)
            workflow.add_node("advanced_question", advanced_question)
            workflow.add_node("groundedness_check", groundedness_check)

            # Connect nodes to each other
            # workflow.add_edge("rewrite_question", "advanced_question")
            # workflow.add_edge("advanced_question", "groundedness_check")
            workflow.add_edge("rewrite_question", "ai_advanced_agent")
            workflow.add_edge("ai_advanced_agent", "advanced_question")
            workflow.add_edge("advanced_question", "groundedness_check")
            # If statement
            workflow.add_conditional_edges(
                "groundedness_check",
                is_grounded,
                {
                    "grounded": END,  # Finish
                    "not grounded": "rewrite_question",  # Rewrite question
                    "not sure": "rewrite_question",  # Rewrite question
                },
            )

            workflow.set_entry_point("rewrite_question")
            memory = MemorySaver()
            app = workflow.compile(checkpointer=memory)
            config = RunnableConfig(
                recursion_limit=5, configurable={"thread_id": "CORRECTIVE-SEARCH-RAG"}
            )
            #   Draw a diagram describing reasoning flow
            # try:
            #     graph = app.get_graph(xray=True)
            #     # Using draw_mermaid_png to render the graph
            #     png_bytes = graph.draw_mermaid_png()
            #     if png_bytes:
            #         # Display the graph
            #         display(Image(png_bytes))
            #     else:
            #         print("No PNG bytes generated")
            # except Exception as e:
            #     print(f"An error occurred: {e}")

            last_output = None

            try:
                for output in app.stream(chat_state, config):
                    last_output = output
                    for key, value in output.items():
                        pprint.pprint(f"\nOutput from node '{key}':")
                        pprint.pprint("---")
                        pprint.pprint(value, indent=2, width=80, depth=None)
            except GraphRecursionError as e:
                print(f"Recursion limit reached: {e}")

            _ = summarize_final_answer(chat_state, last_output)
            print("\n\n----------------Answering routine 2---------------------\n\n", _)

            return _


# Example usage
if __name__ == "__main__":
    chat_history = []  # Initialize chat history
    question_1 = "名古屋市の山本幸司さんがCEOをやっているSES会社に育児休暇はありますか"
    answer_1 = chat(question_1, chat_history, "Gemini_1.5_Flash", "FundastA_社員")


# %%
