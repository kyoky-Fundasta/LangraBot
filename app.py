# %%

from module.llm.advanced_agent import ai_advanced_agent


from data.const import GraphState
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


# Main program, associated with streamlit GUI.
def chat(user_question, chat_history, model_name, who):

    # Main GraphState
    chat_state = GraphState(
        selected_model=model_name,
        question=user_question,
        context=None,
        web=None,
        answer="",
        relevance=None,
        chat_history=chat_history,
        hint=None,
        rewrotten_question=None,
        rewrotten_question_answer=None,
        reasoning=None,
        source=None,
        response_type=None,
    )

    print("------First state : ", chat_state)

    # The guest only can use normal LLM fuction
    if who == "Guest":
        final_answer = normal_question(chat_state)
        print("\n\n------------Normal mode :", type(final_answer), final_answer)
        return final_answer

    # Members can use RAG & Onlin search features.
    elif who == "FundastA_社員":
        chat_state_agent = ai_agent(chat_state)
        print(
            "\n\n-----------received state at app.py :",
            type(chat_state_agent),
            chat_state_agent,
        )

        if chat_state_agent["context"] == None and chat_state_agent["web"] == None:
            # chat_state["response_type"] = "0"
            print(
                "\n\n---------Routine 0------------\n\n",
                type(chat_state_agent),
                chat_state_agent,
            )
            return chat_state_agent
        else:
            print(
                "\n\n---------Routine 1------------\n\n",
                type(chat_state_agent),
                chat_state_agent,
            )
            chat_state_agent_checked = groundedness_check(chat_state_agent)
            print(
                "\n\n-------------checked data :",
                type(chat_state_agent_checked),
                chat_state_agent_checked,
            )
        if chat_state_agent_checked["relevance"] == "grounded":
            # chat_state["response_type"] = "1"
            print(
                f'\n\n----------------Entering routine 1 : {chat_state_agent_checked["relevance"]}---------------------\n\n',
                chat_state_agent_checked,
            )
            return chat_state_agent_checked
        elif (
            chat_state_agent_checked["relevance"] == "not grounded"
            or chat_state_agent_checked["relevance"] == "not sure"
        ):
            # chat_state["response_type"] = "-1"
            print(
                "\n\n---------Entering Routine -1------------\n\n",
                chat_state_agent_checked,
            )

            return chat_state_agent_checked

        elif chat_state["relevance"] == "dummy process":
            print(
                f'\n\n----------------Entering routine 2 : {chat_state["relevance"]}---------------------\n\n',
                chat_state,
            )
            workflow = StateGraph(GraphState)

            # create nodes
            workflow.add_node("rewrite_question", rewrite_question)
            workflow.add_node("ai_advanced_agent", ai_advanced_agent)
            workflow.add_node("advanced_question", advanced_question)
            workflow.add_node("groundedness_check", groundedness_check)

            # Connect nodes to each other
            workflow.add_edge("rewrite_question", "ai_advanced_agent")
            workflow.add_edge("ai_advanced_agent", "advanced_question")
            workflow.add_edge("advanced_question", "groundedness_check")

            # If statement like behavior
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
                recursion_limit=2, configurable={"thread_id": "CORRECTIVE-SEARCH-RAG"}
            )

            ##   Draw a diagram describing reasoning flow (You need jupyter for this function)
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
                        pprint.pprint(f"\n\n\n------------  Output from node '{key}':")
                        pprint.pprint("\n")
                        pprint.pprint(value, indent=2, width=80, depth=None)
            except GraphRecursionError as e:
                print(f"Recursion limit reached: {e}")

            chat_state["response_type"] = 0
            print("\n\n----------------Answering routine 2---------------------\n\n")

            return 0


# Example usage
if __name__ == "__main__":
    chat_history = []  # Initialize chat history
    question_1 = "名古屋市の山本幸司さんがCEOをやっているSES会社に育児休暇はありますか"
    answer_1 = chat(question_1, chat_history, "Gemini_1.5_Flash", "FundastA_社員")


# %%
