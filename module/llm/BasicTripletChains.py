from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from data.const import (
    env_genai,
    env_openai,
    gpt_model_name,
    gemini_model_name,
    GraphState,
)


def gemini_str_parser(model_output):
    return model_output.content.strip().replace("\n", "")


def gemini_chain(prompt, state: GraphState, input) -> str:

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=env_genai,
        temperature=0,
        convert_system_message_to_human=True,
    )
    chain = prompt | model | gemini_str_parser
    response = chain.invoke(input)
    print(response)
    return response


def gpt_chain(prompt, state: GraphState, input) -> str:
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=env_openai)
    chain = prompt | model | StrOutputParser()
    response = chain.invoke(input)

    return response
