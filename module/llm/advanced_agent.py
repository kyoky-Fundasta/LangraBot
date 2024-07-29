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
    gpt_mini_model_name,
)
from module.vector.pineconeDB import FundastA_Policy
from module.web.tavily import web_search
from data.const import env_genai
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from data.prompt_templates.advanced_agent_template import prompt_template
from data.prompt_templates.default_template import prompt_template as default_template


class DynamicPromptCallback(BaseCallbackHandler):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.fundasta_used = False
        self.tool_outputs = []
        self.tool_counter = 1

    def update_prompt(self):
        self.agent.get_prompts()[0].partial_variables[
            "tools"
        ] = "web_search: ウェブの情報を検索してユーザーの質問に答えることが出来ます。\n        以下の時にはこのツールを使ってください。\n        １．Fundasta_policyツールを使ったがユーザーの質問に答えることが出来なかった。\n        ２．ユーザーがリアルタイム情報について質問をした。\n        ３．ユーザーがLLMがまだ学習をしていない最近の情報について質問をした。\n        ４．他のツールでユーザーの質問に答えることが出来なかった時、最後にこのツールを使ってみてください\n        "
        self.agent.get_prompts()[0].partial_variables["tool_names"] = "web_search"
        print("\n----------------allowed tool list updated------------------\n")
        # print(
        #     "New prompt:\n" + self.agent.get_prompts()[0].partial_variables["tools"],
        #     "\ntool_names:",
        #     self.agent.get_prompts()[0].partial_variables["tool_names"],
        # )

    def update_action(self, action):
        print("\n----------------Agent Action modified-------------------\n")
        action.tool = "web_search"
        input = action.tool_input
        action.log = f"Thought_2: I was not able to find relevant information from the FundastA_Policy tool's output. Then, try 'web_search' tool. If you find relevant information from web_search output, use it to generate the answer.\n\nAction: web_search\n\nAction Input: {input}\n"

    def on_agent_action(self, action, **kwargs):
        # print(
        #     "\n\nOld prompt:\n"
        #     + self.agent.get_prompts()[0].partial_variables["tools"],
        #     "\ntool_names:",
        #     self.agent.get_prompts()[0].partial_variables["tool_names"],
        # )
        # print(f"\nAgent action: {action}")

        if action.tool == "FundastA_Policy" and not self.fundasta_used:
            self.fundasta_used = True
            # self.update_prompt()
        elif action.tool == "FundastA_Policy" and self.fundasta_used:
            self.update_action(action)

    def on_tool_end(self, output: str, **kwargs) -> None:

        print(f"\n\n-----------#{self.tool_counter} : Tool ended----------------\n")
        self.tool_counter += 1
        # print(f"\n\n{kwargs['name']} Tool ended. Output: {output}")
        self.tool_outputs.append({kwargs["name"]: output})


def ai_advanced_agent(chat_state: GraphState) -> GraphState:
    selected_model = chat_state["selected_model"]
    chat_history_str = (
        "\n".join([f"ユーザー: {q}\nAI: {a}" for q, a in chat_state["chat_history"]])
        + chat_state["question"]
    )
    chat_history_str = (
        chat_history_str
        + f'\nユーザー:{chat_state["question"]}\nAI:{chat_state["answer"]},{chat_state["reasoning"]}'
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

    prompt = default_template
    prompt.template = prompt_template

    dynamic_callback = DynamicPromptCallback(None)
    tools = [
        FundastA_Policy(callbacks=[dynamic_callback]),
        web_search(callbacks=[dynamic_callback]),
    ]
    agent = create_react_agent(llm, tools, prompt=prompt)
    dynamic_callback.agent = agent

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
    chat_state["rewrotten_question_answer"] = agent_executor.invoke(
        {
            "rewrotten_question": chat_state["rewrotten_question"],
            "history": chat_history_str,
        }
    )

    chat_state["hint"] = (
        chat_state["hint"]
        + " - "
        + chat_state["rewrotten_question"]
        + " : "
        + chat_state["rewrotten_question_answer"]
        + "\n"
    )
    for output in dynamic_callback.tool_outputs:
        if "FundastA_Policy" in output:
            chat_state["context"] = chat_state["context"] + output["FundastA_Policy"]
        elif "web_search" in output:
            chat_state["web"] = chat_state["web"] + output["web_search"]
    print(chat_state)

    return chat_state


if __name__ == "__main__":
    # user_input = "FundastAの有給休暇について説明してください"
    user_input = "名古屋市にある山本幸次さんがCEOをやっている会社の住所はどこですか"
    rewrotten_input = "質問しているの会社の名前はFundastAです。"
    # user_input = "こんにちは、世界で一番高いビルは何ですか"
    # user_input = input("Question :")

    # model_name = "Gemini_1.5_Flash"
    # model_name = "ChatGPT_3.5"
    model_name = "ChatGPT_4o_mini"

    test_state = GraphState(
        selected_model=model_name,
        question=user_input,
        context="",
        web="",
        answer="",
        relevance="",
        chat_history=[],
        hint="",
        rewrotten_question=rewrotten_input,
        rewrotten_question_answer="",
        reasoning="",
    )

    ai_advanced_agent(test_state)
