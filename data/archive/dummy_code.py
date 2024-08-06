def summarize_final_answer(chat_state):
    final_response = {
        "question": "",
        "answer": "",
        "relevance": "",
        "reasoning": "",
        "source": "",
    }
    if chat_state["relevance"] == "grounded":
        final_response["question"] = chat_state["question"]
        final_response["answer"] = chat_state["answer"]
        final_response["relevance"] = chat_state["relevance"]
        final_response["reasoning"] = chat_state["reasoning"]
        final_response["source"] = chat_state["source"]
    else:
        final_response["question"] = chat_state["question"]
        final_response["answer"] = (
            "申し訳ありません。詳しい情報が見つかりませんでした。担当部署にお問い合わせください。"
        )
        final_response["relevance"] = chat_state["relevance"]
        final_response["reasoning"] = chat_state["reasoning"]
        final_response["source"] = chat_state["source"]
    return final_response
