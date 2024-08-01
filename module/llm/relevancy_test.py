from data.const import (
    llm_switch,
    GraphState,
    gemini_model_name,
    gpt_mini_model_name,
    gpt_model_name,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from data.test_data.test_state import test_result_0, test_result_1
from data.prompt_templates.relevancy_checker_template import prompt_template


class check_output(BaseModel):
    result: str = Field(
        description="It should be on of  ['grounded', 'not grounded', 'not sure']"
    )
    reasoning: str = Field(
        description="Explain the reason for the result shortly in Japanese"
    )
    source: str = Field(description="Provide source & page or source URL")


# return groundedness result in json format : result, reasoning, source
def groundedness_check(input_state: GraphState) -> GraphState:
    selected_model = input_state["selected_model"]
    if not (input_state["context"] and input_state["web"]):
        input_state["relevance"] = ""
        return input_state

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
    llm = llm_switch(selected_model)

    chain = prompt | llm | parser
    result_json = chain.invoke({"question": input_state["question"]})
    input_state["relevance"] = result_json["result"]
    input_state["reasoning"] = result_json["reasoning"]
    input_state["source"] = result_json["source"]
    print(
        "\n\n------------------Double check result :",
        type(result_json),
        "\n",
        result_json,
    )
    return input_state


def is_grounded(result_json):

    return result_json["relevance"]


if __name__ == "__main__":

    gemini = groundedness_check(gemini_model_name, test_result_1)
    print("\n\n------------------interval--------------------\n\n")
    mini = groundedness_check(gpt_mini_model_name, test_result_1)
    print("\n\n------------------interval--------------------\n\n")
