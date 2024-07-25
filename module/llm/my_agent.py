# Fixed the agent's inappropriate behavir by AgentExecutor's callback parameter.
# Check the line no.92 and DynamicPromptCallback class

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from data.const import (
    env_genai,
    gemini_model_name,
    GraphState,
    env_openai,
    gpt_model_name,
    agent_prompt_mod,
)
from module.vector.pineconeDB import FundastA_Policy
from module.web.tavily import web_search
from data.const import env_genai
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler


class DynamicPromptCallback(BaseCallbackHandler):
    def __init__(self, agent):
        print("DynamicPromptCallback initialized")
        self.agent = agent
        self.fundasta_used = False

    def update_prompt(self):
        print("\n\n-----------Trigger 2-------------\n\n")
        self.agent.get_prompts()[0].partial_variables[
            "tools"
        ] = "web_search: ウェブの情報を検索してユーザーの質問に答えることが出来ます。\n        以下の時にはこのツールを使ってください。\n        １．Fundasta_policyツールを使ったがユーザーの質問に答えることが出来なかった。\n        ２．ユーザーがリアルタイム情報について質問をした。\n        ３．ユーザーがLLMがまだ学習をしていない最近の情報について質問をした。\n        ４．他のツールでユーザーの質問に答えることが出来なかった時、最後にこのツールを使ってみてください\n        "
        self.agent.get_prompts()[0].partial_variables["tool_names"] = "web_search"
        print(
            "New prompt:\n" + self.agent.get_prompts()[0].partial_variables["tools"],
            "\ntool_names:",
            self.agent.get_prompts()[0].partial_variables["tool_names"],
        )

    def update_action(self, action):
        print("\n----------------Action modification-------------------\n")
        action.tool = "web_search"
        input = action.tool_input
        action.log = f"Thought_2: I was not able to find relevant information from the FundastA_Policy tool's output. Then, try 'web_search' tool. If you find relevant information from web_search output, use it to generate the answer.\n\nAction: web_search\n\nAction Input: {input}\n"

    def on_agent_action(self, action, **kwargs):
        print(
            "\n\nOld prompt:\n"
            + self.agent.get_prompts()[0].partial_variables["tools"],
            "\ntool_names:",
            self.agent.get_prompts()[0].partial_variables["tool_names"],
        )
        print(f"\nAgent action: {action}")

        if action.tool == "FundastA_Policy" and not self.fundasta_used:
            print("\n\n-----------Trigger 1-------------\n\n")
            self.fundasta_used = True
            self.update_prompt()
        elif action.tool == "FundastA_Policy" and self.fundasta_used:
            self.update_action(action)


def ai_agent(selected_model, chat_state: GraphState) -> GraphState:

    if selected_model == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=gemini_model_name,
            google_api_key=env_genai,
            temperature=0,
            convert_system_message_to_human=True,
        )
    elif selected_model == "gpt":
        llm = ChatOpenAI(temperature=0, model=gpt_model_name, openai_api_key=env_openai)
    # When creating the AgentExecutor:
    tools = [FundastA_Policy, web_search]
    prompt = hub.pull("hwchase17/react")
    prompt.template = agent_prompt_mod
    tools = [FundastA_Policy(), web_search()]
    agent = create_react_agent(llm, tools, prompt=prompt)
    dynamic_callback = DynamicPromptCallback(agent)
    print("Callback created")
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=3,
        return_intermediate_steps=False,
        callbacks=[dynamic_callback],
        early_stopping_method="generate",
    )
    # chat_state["answer"] = agent_executor.invoke({"question": chat_state["question"]})
    # print("\nOutput State by agent :\n" + chat_state)
    output = agent_executor.invoke({"question": user_input})
    print(output, type(output))

    return chat_state


if __name__ == "__main__":
    # user_input = "FundastAの有給休暇について説明してください"
    user_input = "FundastAの社員数は何人ですか"
    # user_input = "こんにちは、世界で一番高いビルは何ですか"
    # user_input = input("Question :")
    model_name = "gemini"
    ai_agent(model_name, user_input)
