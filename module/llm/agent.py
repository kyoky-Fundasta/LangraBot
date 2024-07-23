from langchain import hub
from langchain_community.llms import OpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from data.const import (
    env_genai,
    gemini_model_name,
    GraphState,
    env_openai,
    gpt_model_name,
    agent_prompt_mod,
)
from module.vector.pineconeDB import retrieve_document, FundastA_Policy
from module.web.tavily import web_search
from module.web.tavily import search_on_web
from data.const import env_genai
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


# gemini_model_name = "gemini-1.5-flash"
test_state = GraphState(
    question="",
    context="",
    web="",
    answer="",
    relevance="",
    chat_history=[],
)


def ai_agent(selected_model, user_input):

    if selected_model == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=gemini_model_name,
            google_api_key=env_genai,
            temperature=0,
            convert_system_message_to_human=True,
        )
    elif selected_model == "gpt":
        llm = ChatOpenAI(temperature=0, model=gpt_model_name, openai_api_key=env_openai)

    tools = [FundastA_Policy(), web_search()]
    # tools = [web_search()]
    prompt = hub.pull("hwchase17/react")
    prompt.template = agent_prompt_mod
    # print(prompt)
    agent = create_react_agent(llm, tools, prompt=prompt)
    # print(agent)
    agent_excutor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=3,
    )

    # user_input = input("Enter your question :")

    output = agent_excutor.invoke({"input": user_input})
    print(output, type(output))


if __name__ == "__main__":
    # user_input = "FundastAの有給休暇について説明してください"
    # user_input = "FundastAの住所はどこですか"
    user_input = "こんにちは、世界で一番高いビルは何ですか"
    # user_input = input("Question :")
    model_name = "gpt"
    ai_agent(model_name, user_input)
