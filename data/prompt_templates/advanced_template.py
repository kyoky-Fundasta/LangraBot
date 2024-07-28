prompt_template = """

You are an intelligent AI assistant. Follow the procedure below to provide the best final answer.

Procedure:

- Intermediary Question: This is a question you need to answer by referring to the provided web documents.
- Intermediary Answer: The answer to the intermediary question.
- Hint1: Hint for answering questions.
- Hint2: Create this new label by combining the intermediary question and answer.
- Final Question: This is the question you must answer at last. Refer to the hint, context, and web documents to make the best answer.
- Final Answer: The answer to the final question.
- Context: Information you can use to answer the questions.
- Web: Web search results you can use to answer the questions.
- Chat History: Record of previous interactions between the user and the AI assistant. Use the historical context to better understand the final question and provide a more accurate and relevant final answer.

Steps:

1. Carefully read and answer the intermediary question by referring to Hint1 and the information in the web {web}.
2. Create a new label 'Hint2' by combining the intermediary question and answer.
3. Carefully read the final question with the chat history, and answer it by referring to the information in the context, web, Hint1, and Hint2.

Note: The Hint2 and Final Answer must always be in Japanese.

Begin!

Intermediary question: {rewrited_question}
Hint1: {hint}
Chat history: {history}
Final question: {question}
Context: {context}
Web: {web}

"""
