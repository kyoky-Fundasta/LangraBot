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

    prompt_template = """

You are a judge of an AI assistant.
You should evaluate the groundedness of the AI's answer to a given question.
Use the historical context to better understand the current question and provide a more accurate evaluation.

Chat history: record of previous interactions between the user and the AI assistant.
question: the question the AI should answer.
answer: the answer the AI generated.
context: the source information from documents used by the AI to generate the correct answer.
web: the source information from websites used by the AI to generate the correct answer.

Your task is to judge the groundedness of the AI's answer for the question.
Result: You should judge the groundedness of the answer (not the source information) based on the question.
Reasoning: Provide a concise explanation of the result.
Source: Cite the source and page (for context documents) or source URL (for web documents) used to answer the question.

Use one of the following patterns to provide your result:

- Pattern 1: not grounded - If the AI failed to answer the question (e.g., "Agent stopped......", "AI could not find any relevant information", "I do not know", " ", or similar), you must judge it as "not grounded" even if the documents contain the correct answer.
- Pattern 2: not grounded - If the AI's answer is opposed to or different from what is written in the source information.
- Pattern 3: grounded - If the AI clearly answers the question and the answer is grounded in any of the documents.
- Pattern 4: not sure - If you are unable to judge whether the AI's answer is grounded or not.

Let's carefully study some examples:

# Example 1:
question: "What is the tallest mountain in Japan?"
answer: "AI stopped due to the limitation" or "I was not able to find any relevant information" or "I do not know" or "" or similar
web: "<content>Mount Fuji is the tallest mountain in Japan.</content><source>https://test.aa.bb/</source>"

result: not grounded
reasoning: The web source provides the correct answer, but the AI failed to answer the question.
source : ""

# Example 2:
question: "What is the tallest mountain in Japan?"
answer: "Japan has many volcanoes and they are tall and beautiful"
context: "<content>Mount Fuji is the tallest mountain in Japan.</content><source>日本の山 PDFファイル</source><page>10</page>"

result: not grounded
reasoning: The answer is true but does not address the user's question.
source : ""

# Example 3:
question: "What is the tallest mountain in Japan?"
answer: "Mount Aino is the tallest mountain in Japan"
web: "<content>Mount Fuji is the tallest mountain in Japan.</content><source>https://test.aa.bb/</source>"

result: not grounded
reasoning: The answer is different from the information in the documents.
source : ""

# Example 4:
question: "What is the tallest mountain in Japan?"
answer: "Mount Fuji"
context: "<content>Mount Fuji is the tallest mountain in Japan.</content><source>日本の山 PDFファイル</source><page>10</page>"
web: "<content>Mount Fuji is the tallest mountain in Japan.</content><source>https://test.aa.bb/</source>"

result: grounded
reasoning: The answer addresses the question and is grounded in the source information.
source : "日本の山 PDFファイル:page 10, web:https://test.aa.bb/"

# Now you should judge the real question:
Chat history: {history}
question: {question}
answer: {answer}
context: {context}
web: {web}

your answer:
{format_instructions}


"""

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
