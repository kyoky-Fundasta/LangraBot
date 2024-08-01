# AI assistant agent. Answer a user question using tools.
# StateGraph -> StateGraph

import re
from uuid import UUID

from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Dict, Any, Optional
from typing import Any, Dict, List, Union


from module.vector.pineconeDB import FundastA_Policy
from module.web.tavily import web_search

from data.prompt_templates.agent_template import prompt_template, agent_updated_template
from data.prompt_templates.default_template import prompt_template as default_template
from data.const import (
    llm_switch,
    gemini_model_name,
    GraphState,
    gpt_model_name,
    gpt_mini_model_name,
)


# Callback calss. This is for controlling the AgentExecutor's behavior.


class DynamicPromptCallback(BaseCallbackHandler):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.original_question = None
        self.tool_outputs = []
        self.used_tool_name = []
        self.tool_output_values = []
        self.tool_counter = 1
        self.action_count = 1
        self.llm_comment = ""

    def trim_llm_text(self, llm_text: str) -> str:
        pattern = r"Thought:(.*?)Action:"
        match = re.search(pattern, llm_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return ""

    # A custom function to update agent's prompt.
    # I made this because the default prompt did not work with my agent properly after the second iteration.
    def update_prompt(self):
        self.agent.get_prompts()[0].template = agent_updated_template
        self.agent.get_prompts()[0].partial_variables["used_tool"] = (
            self.used_tool_name[-1] if self.used_tool_name else "None"
        )
        self.agent.get_prompts()[0].partial_variables["tool_output"] = (
            self.tool_output_values[-1] if self.tool_output_values else "None"
        )
        print("\n----------------prompt updated------------------\n")

    ## Updating action is alternative way of controlling agent's behavior.

    # def update_action(self, action, action_code):

    #     if action_code == 1:
    #         print(
    #             "\n----------------Agent Action modified to web_search -------------------\n"
    #         )
    #         action.tool = "web_search"
    #         input = action.tool_input
    #         action.log = f"Thought_2: I was not able to find relevant information from the FundastA_Policy tool's output. Then, try 'web_search' tool. If you find relevant information from web_search output, use it to generate the answer.\n\nAction: web_search\n\nAction Input: {input}\n"

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ):
        if self.original_question is None:
            self.original_question = inputs["question"]
        print(f"\n------------------Chain start : {self.original_question}\n")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        # print(f"\n------------------LLM start\n{serialized}\n")
        print(f"\n------------------LLM start\n\n")

    ## To observe how the agent's llm generates tokens.
    # def on_llm_new_token(
    #     self,
    #     token: str,
    #     *,
    #     chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
    #     run_id: UUID,
    #     parent_run_id: Optional[UUID] = None,
    #     **kwargs: Any,
    # ) -> Any:
    #     print(f"\n------------------New token : {token}\n")

    def on_agent_action(self, action, **kwargs):
        print(f"\n------------------#{self.action_count} Agent action:")

        self.action_count += 1

    # Agent's prompt should be updated before it starts its llm.
    def on_tool_end(self, output: str, **kwargs):

        # print(f"\n\n{kwargs['name']} Tool ended. Output: {output}")
        self.tool_outputs.append({kwargs["name"]: output})
        self.used_tool_name.append(kwargs["name"])
        self.tool_output_values.append(output)
        self.update_prompt()
        print(f"\n\n-----------#{self.tool_counter} : Tool ended----------------\n")
        self.tool_counter += 1

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        # print(f"\n------------------Chain end : {outputs}\n")
        print(f"\n------------------Chain end : \n")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:

        trimmed_comment = self.trim_llm_text(response.generations[0][0].text)
        self.llm_comment = self.llm_comment + trimmed_comment
        print(
            f"\n------------------LLM end : \n{self.llm_comment}\n--------------------"
        )


# This is the first agent the main app calls. I expect most questions should be solved by this.
def ai_agent(chat_state: GraphState) -> GraphState:

    selected_model = chat_state["selected_model"]
    chat_history_str = "\n".join(
        [f"ユーザー: {q}\nAI: {a}" for q, a in chat_state["chat_history"]]
    )
    llm = llm_switch(selected_model)

    # Initiate a callback instance and integrate it with llm, tools and agent.
    dynamic_callback = DynamicPromptCallback(None)
    llm.callbacks = [dynamic_callback]
    prompt = default_template
    # prompt.template = prompt_template

    tools = [
        FundastA_Policy(callbacks=[dynamic_callback]),
        web_search(callbacks=[dynamic_callback]),
    ]
    agent = create_react_agent(
        llm,
        tools,
        prompt=prompt,
    )
    dynamic_callback.agent = agent

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=3,
        return_intermediate_steps=True,
        callbacks=[dynamic_callback],
        early_stopping_method="force",
    )
    input_variables = {"question": chat_state["question"], "history": chat_history_str}

    agent_final_answer = agent_executor.invoke(input_variables)
    chat_state["answer"] = agent_final_answer["output"]
    chat_state["hint"] = (
        chat_state["hint"]
        + "・質問："
        + chat_state["question"]
        + "\n・回答："
        + chat_state["answer"]
        + "\n・解説："
        + dynamic_callback.llm_comment
        + "\n"
    )

    # Add tool's outputs to the state.
    for output in dynamic_callback.tool_outputs:
        if "FundastA_Policy" in output:
            chat_state["context"] = output["FundastA_Policy"]
        elif "web_search" in output:
            chat_state["web"] = output["web_search"]
    # print(chat_state)
    print(chat_state["answer"])

    return chat_state


if __name__ == "__main__":
    user_input = "名古屋市の山本幸司さんがCEOをやっているSES会社に育児休暇はありますか"
    # user_input = "FundastAの住所はどこですか"
    # user_input = "こんにちは、世界で一番高いビルは何ですか"
    # user_input = input("Question :")

    model_name = "Gemini_1.5_Flash"
    # model_name = "ChatGPT_3.5"
    # model_name = "ChatGPT_4o_mini"

    test_state = GraphState(
        selected_model=model_name,
        question=user_input,
        context="",
        web="",
        answer="",
        relevance="",
        chat_history=[],
        hint="",
        rewrotten_question="",
        rewrotten_question_answer="",
        reasoning="",
        source="",
    )

    ai_agent(test_state)
