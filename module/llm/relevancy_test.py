from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from data.const import (
    GraphState,
    gemini_model_name,
    env_genai,
    gpt_mini_model_name,
    env_openai,
    gpt_model_name,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from data.test_data.test_state import test_result_0, test_result_1
from data.prompt_templates.relevancy_checker import prompt_template


class check_output(BaseModel):
    Result: str = Field(
        description="It should be on of  ['grounded', 'not grounded', 'not sure']"
    )
    Reasoning: str = Field(
        description="Explain the reason for the result shortly in Japanese"
    )
    Source: str = Field(description="Provide source & page or source URL")


def groundedness_check(selected_model, input_state: GraphState):
    parser = JsonOutputParser(pydantic_object=check_output)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question"],
        partial_variables={
            "answer": input_state["answer"],
            "context": input_state["context"],
            "web": input_state["web"],
            "history": input_state["chat_history"],
            "format_instructions": parser.get_format_instructions(),
        },
    )

    if selected_model == "Gemini_1.5_Flash":
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=env_genai,
            temperature=0,
            convert_system_message_to_human=True,
        )
        print("Gemini selected")
    elif selected_model == "ChatGPT_3.5":
        llm = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo-0125", openai_api_key=env_openai
        )
        print("GPT selected")

    elif selected_model == "ChatGPT_4o_mini":
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=env_openai)
        print("GPT mini selected")

    chain = prompt | llm | parser
    result_json = chain.invoke({"question": input_state["question"]})

    print(type(result_json), "\n", result_json)
    return result_json


if __name__ == "__main__":

    gemini = groundedness_check(gemini_model_name, test_result_1)
    print("\n\n------------------interval--------------------\n\n")
    mini = groundedness_check(gpt_mini_model_name, test_result_1)
    print("\n\n------------------interval--------------------\n\n")
