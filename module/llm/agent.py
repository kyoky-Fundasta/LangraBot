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

# llm = ChatGoogleGenerativeAI(
#     model=gemini_model_name,
#     google_api_key=env_genai,
#     temperature=0,
#     convert_system_message_to_human=True,
# )

llm = ChatOpenAI(temperature=0, model=gpt_model_name, openai_api_key=env_openai)

tools = [FundastA_Policy()]

prompt = hub.pull("hwchase17/react")
prompt.template = agent_prompt_mod


agent = create_react_agent(llm, tools, prompt=prompt)

agent_excutor = AgentExecutor(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True, max_iterations=2
)

# user_input = input("Enter your question :")
user_input = "FundastAの有給休暇について説明してください"
output = agent_excutor.invoke({"input": user_input})
print(output, type(output))
