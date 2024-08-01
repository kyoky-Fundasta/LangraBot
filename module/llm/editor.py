from data.const import GraphState
from langchain_core.prompts import ChatPromptTemplate
from data.prompt_templates.rewriter_template import prompt_template
from langchain_core.output_parsers import StrOutputParser
from data.const import llm_switch


# Rewrite the user question. Returns a new GraphState.
def rewrite_question(chat_state: GraphState) -> GraphState:
    selected_model = chat_state["selected_model"]
    chat_history_str = "\n".join(
        [f"ユーザー: {q}\nAI: {a}" for q, a in chat_state["chat_history"]]
    )
    question = chat_state["question"]
    context = chat_state["context"]
    web = chat_state["web"]
    answer = chat_state["answer"]
    reasoning = chat_state["reasoning"]

    prompt = prompt_template

    llm = llm_switch(selected_model)

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(
        {
            "chat_history": chat_history_str,
            "question": question,
            "answer": answer,
            "reasoning": reasoning,
            "context": context,
            "web": web,
        }
    )
    chat_state["rewrotten_question"] = response
    print("\n------updated question :", chat_state)
    return chat_state
