from langchain_core.prompts import ChatPromptTemplate


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional prompt rewriter. Your task is to generate the question in order to get additional information that is now shown in the context."
            "Your generated question will be searched on the web to find relevant information.",
        ),
        (
            "human",
            "Rewrite the question to get additional information to get the answer."
            "\n\nHere is the chat history:\n ------- \n{chat_history}\n ------- \n"
            "\n\nHere is the initial question:\n ------- \n{question}\n ------- \n"
            "\n\nHere is the initial context:\n ------- \n{context}\n ------- \n"
            "\n\nHere is the initial online search result:\n ------- \n{web}\n ------- \n"
            "\n\nHere is the initial answer to the question:\n ------- \n{answer}\n ------- \n"
            "\n\nFormulate an improved question in Japanese:",
        ),
    ]
)
