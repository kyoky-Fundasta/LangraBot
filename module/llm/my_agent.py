# Fixed the agent's inappropriate behavir by AgentExecutor's callback parameter.
# Check the line no.92 and DynamicPromptCallback class


from uuid import UUID
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
    create_tool_calling_agent,
)
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult
from data.const import (
    llm_switch,
    gemini_model_name,
    GraphState,
    gpt_model_name,
    gpt_mini_model_name,
)
from module.vector.pineconeDB import FundastA_Policy
from module.web.tavily import web_search
from module.llm.get_response import answering_bot
from langchain.callbacks.base import BaseCallbackHandler
from data.prompt_templates.agent_template import prompt_template, agent_updated_template
from data.prompt_templates.default_template import prompt_template as default_template
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, Optional
from typing import Any, Dict, List, Union

# from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_core.agents import AgentAction


class DynamicPromptCallback(BaseCallbackHandler):
    def __init__(self, agent, llm):
        super().__init__()
        self.agent = agent
        self.llm = llm
        self.fundasta_used = False
        self.web_used = False
        self.original_question = None
        self.tool_outputs = []
        self.used_tool_name = []
        self.tool_output_values = []
        self.tool_counter = 1
        self.action_count = 0

    def update_prompt(self):
        self.agent.get_prompts()[0].template = agent_updated_template
        # self.agent.get_prompts()[0].partial_variables[
        #     "tools"
        # ] = "web_search: ウェブの情報を検索してユーザーの質問に答えることが出来ます。\n        以下の時にはこのツールを使ってください。\n        １．Fundasta_policyツールを使ったがユーザーの質問に答えることが出来なかった。\n        ２．ユーザーがリアルタイム情報について質問をした。\n        ３．ユーザーがLLMがまだ学習をしていない最近の情報について質問をした。\n        ４．他のツールでユーザーの質問に答えることが出来なかった時、最後にこのツールを使ってみてください\n        "
        # self.agent.get_prompts()[0].partial_variables["tool_names"] = "web_search"
        self.agent.get_prompts()[0].partial_variables["used_tool"] = (
            self.used_tool_name[-1] if self.used_tool_name else "None"
        )
        self.agent.get_prompts()[0].partial_variables["tool_output"] = (
            self.tool_output_values[-1] if self.tool_output_values else "None"
        )
        print("\n----------------allowed tool list updated------------------\n")

    # print(
    #     "New prompt:\n" + self.agent.get_prompts()[0].partial_variables["tools"],
    #     "\ntool_names:",
    #     self.agent.get_prompts()[0].partial_variables["tool_names"],
    # )

    def update_action(self, action, action_code):

        if action_code == 1:
            print(
                "\n----------------Agent Action modified to web_search -------------------\n"
            )
            action.tool = "web_search"
            input = action.tool_input
            action.log = f"Thought_2: I was not able to find relevant information from the FundastA_Policy tool's output. Then, try 'web_search' tool. If you find relevant information from web_search output, use it to generate the answer.\n\nAction: web_search\n\nAction Input: {input}\n"

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ):
        if self.original_question is None:
            self.original_question = inputs["question"]
        print(f"\n------------------Chain start : {self.original_question}\n")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        print(f"\n------------------LLM start\n {serialized}:\n")

        # prompts[0] = "Just say 'Hello'"

        print("------------------------")

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
        # print(
        #     "\n\nOld prompt:\n"
        #     + self.agent.get_prompts()[0].partial_variables["tools"],
        #     "\ntool_names:",
        #     self.agent.get_prompts()[0].partial_variables["tool_names"],
        # )
        print(f"\n------------------Agent action: \n{action.log}\n")

        # if self.action_count > 0:
        #     # Force the agent to consider the last tool output
        #     forced_thought = f"""I have reviewed the tool output. If relevant information was found, I will generate a final answer and end the chain.
        #     If no relevant information was found, I will select another tool or search for information using a different approach.
        #     Last tool output: {self.tool_outputs[-1]}
        #     Based on this information, please determine the next action."""
        #     action.log = f"{action.log}\n\nThought: {forced_thought}"
        self.action_count += 1
        # self.agent.action = action   ---error

        # if action.tool == "FundastA_Policy" and not self.fundasta_used:
        #     self.fundasta_used = True
        # elif action.tool == "FundastA_Policy" and self.fundasta_used:
        #     self.update_action(action, 1)
        # elif action.tool == "web_search" and not self.web_used:
        #     self.web_used = True
        # elif action.tool == "web_search" and self.web_used:
        #     self.update_action(action, 2)

    def on_tool_end(self, output: str, **kwargs):

        print(f"\n\n-----------#{self.tool_counter} : Tool ended----------------\n")

        self.tool_counter += 1
        # print(f"\n\n{kwargs['name']} Tool ended. Output: {output}")
        self.tool_outputs.append({kwargs["name"]: output})
        self.used_tool_name.append(kwargs["name"])
        self.tool_output_values.append(output)
        self.update_prompt()
        # prompt = PromptTemplate(
        #     template="""Based on the following information from a tool and the original question, determine if you can generate a final answer or if you need more information.

        # Original question: {question}

        # Tool output: {output}

        # Instructions:
        # 1. If you can provide a reliable answer based on this information, generate the final answer in Japanese.
        # 2. If you cannot provide a reliable answer, clearly state that you need more information and specify what kind of information you need.

        # Your response:
        # """,
        #     input_variables=["question", "output"],
        # )
        # chain = prompt | self.llm | StrOutputParser
        # chain.invoke({"question": self.original_question, "output": output})
        # chain.invoke({"question": "日本の首都はどこ", "output": output})

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        print(f"\n------------------Chain end : {outputs}\n")

    #     if "intermediate_steps" in outputs:
    #         last_action = outputs["intermediate_steps"][-1][0]
    #         last_observation = outputs["intermediate_steps"][-1][1]
    #         if "I now know the final answer" not in last_action.log:
    #             forced_thought = f"""I have reviewed the result of the last action. If I can generate a final answer to the question, I will do so.
    # If not, I will consider a different approach.
    # Last action: {last_action.log}
    # Last observation: {last_observation}
    # Based on this information, please generate a final answer or determine the next step."""

    #             # Use the agent's chain to predict instead of directly accessing an llm
    #             outputs["output"] = self.agent({"input": forced_thought})["output"]

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        print(
            f"\n------------------LLM end : \n{response.generations[0][0].text}\n--------------------"
        )


def ai_agent(chat_state: GraphState) -> GraphState:
    selected_model = chat_state["selected_model"]
    chat_history_str = "\n".join(
        [f"ユーザー: {q}\nAI: {a}" for q, a in chat_state["chat_history"]]
    )
    llm = llm_switch(selected_model)
    dynamic_callback = DynamicPromptCallback(None, llm)
    llm.callbacks = [dynamic_callback]
    prompt = default_template
    prompt.template = prompt_template

    tools = [
        FundastA_Policy(callbacks=[dynamic_callback]),
        web_search(callbacks=[dynamic_callback]),
        answering_bot(callbacks=[dynamic_callback]),
    ]
    agent = create_react_agent(llm, tools, prompt=prompt)
    # agent = create_tool_calling_agent(llm, tools, prompt=prompt)
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
    for output in dynamic_callback.tool_outputs:
        if "FundastA_Policy" in output:
            chat_state["context"] = output["FundastA_Policy"]
        elif "web_search" in output:
            chat_state["web"] = output["web_search"]
    print(chat_state)

    return chat_state


if __name__ == "__main__":
    # user_input = "FundastAの有給休暇について説明してください"
    user_input = "FundastAの住所はどこですか"
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
