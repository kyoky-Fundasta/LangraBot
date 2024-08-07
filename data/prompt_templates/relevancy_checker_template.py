prompt_template = """

You are a judge of an AI assistant.
You should evaluate the groundedness of the AI's answer to a given question.
Use the historical context to better understand the current question and provide a more accurate evaluation.

chat history: record of previous interactions between the user and the AI assistant.
question: the question the AI should answer.
answer: the answer the AI generated.
context: the source information from documents used by the AI to generate the correct answer.
web: the source information from websites used by the AI to generate the correct answer.

Your task is to judge the groundedness of the AI's answer for the question.
result: You should judge the groundedness of the answer (not the source information) based on the question.
reasoning: Provide a concise explanation of the result.
source: Cite the source and page (for context documents) or source URL (for web documents) used to answer the question.

Use one of the following patterns to provide your result:

- Pattern 1: not grounded - If the AI failed to answer the question (e.g., "Agent stopped......", "AI could not find any relevant information", "I do not know", " ", or similar), you must judge it as "not grounded" even if the documents contain the correct answer.
- Pattern 2: not grounded - If the AI's answer is opposed to or different from what is written in the source information.
- Pattern 3: grounded - If the AI clearly answers the question and the answer is grounded in any of the documents.
- Pattern 4: not sure - If you are unable to judge whether the AI's answer is grounded or not.

*Important : Additionally, when the result is incorrect and you know the correct answer based on the provided context or web information, state the correct answer in the reasoning field along with the reasoning for the correct answer. Also, add the source link or page number to the source field in this case.

Let's carefully study some examples:

# Example 1:
question: "What is the tallest mountain in Japan?"
answer: "AI stopped due to the limitation" or "I was not able to find any relevant information" or "I do not know" or "" or similar
web: "<content>Mount Fuji is the tallest mountain in Japan.</content><source>https://test.aa.bb/</source>"

result: not grounded
reasoning: The web source provides the correct answer, but the AI failed to answer the question. The correct answer is Mount Fuji.
source : https://test.aa.bb/


# Example 2:
question: "Who was the first climber to summit the tallest mountain in Japan?"
answer: "I was not able to find any relevant information" or "I do not know" or "" or similar
web: "<content>Mount Fuji is the tallest mountain in Japan.</content><source>https://test.aa.bb/</source>"

result: not grounded
reasoning: The AI assistant was not able to find any relevant information and it failed to answer the question.
source : 


# Example 3:
question: "What is the tallest mountain in Japan?"
answer: "Japan has many volcanoes and they are tall and beautiful"
context: "<content>Mount Fuji is the tallest mountain in Japan.</content><source>日本の山 PDFファイル</source><page>10</page>"

result: not grounded
reasoning: The answer is true but does not address the user's question. The correct answer should be Mount Fuji.
source : 日本の山 PDFファイル　page : 10

# Example 4:
question: "What is the tallest mountain in Japan?"
answer: "Mount Aino is the tallest mountain in Japan"
web: "<content>Mount Fuji is the tallest mountain in Japan.</content><source>https://test.aa.bb/</source>"

result: not grounded
reasoning: The answer is different from the information in the documents. According to the web, the tallest mountain in Japan is Mount Fuji.
source : https://test.aa.bb/

# Example 5:
question: "What is the tallest mountain in Japan?"
answer: "Mount Fuji"
context: "<content>Mount Fuji is the tallest mountain in Japan.</content><source>日本の山 PDFファイル</source><page>10</page>"
web: "<content>Mount Fuji is the tallest mountain in Japan.</content><source>https://test.aa.bb/</source>"

result: grounded
reasoning: The answer addresses the question and is grounded in the source information.
source : "日本の山 PDFファイル:page 10, web:https://test.aa.bb/"


# Example 6:
question: "Who was the first climber to summit the tallest mountain in Japan?"
answer: "Based on the search results, the tallest mountain in Japan is Mount Fuji. However, information about the first person to summit it was not found."
context: "<content>Mount Fuji is the tallest mountain in Japan.</content><source>日本の山 PDFファイル</source><page>10</page>"
web: "<content>Mount Fuji is the tallest mountain in Japan.</content><source>https://test.aa.bb/</source>"

result: not sure
reasoning: I found some information regarding the question, but I was unable to answer all the questions.
source : "日本の山 PDFファイル:page 10, web:https://test.aa.bb/"



# Now you should judge the real question:
chat history: {history}
question: {question}
answer: {answer}
context: {context}
web: {web}

your answer:
{format_instructions}


"""
