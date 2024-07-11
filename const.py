import os
from dotenv import load_dotenv
from typing import TypedDict

home_directory = os.path.expanduser("~")
env_folder = os.path.join(home_directory, ".openai")
env_path = os.path.join(env_folder, ".env")
_ = load_dotenv(env_path)

env_tavily = os.getenv("Tavily_API_KEY_An")
env_openai = os.getenv("OPENAI_API_KEY_FundastA")
env_genai = os.getenv("Gemini_API_KEY_An")
env_smith = os.getenv("LANGCHAIN_API_KEY")
embedding_model = "text-embedding-3-small"
index_name = "fundasta"


# GraphState is a TypedDict that defines the structure of the  graph state
class GraphState(TypedDict):
    question: str
    context: str
    web: str
    answer: str
    relevance: str
    chat_history: list


# state = {
#     "question": "What is the capital of France?",
#     "context": "The capital of France is Paris.",
#     "web": "",
#     "answer": "",
#     "relevance": "",
# }

# if __name__ == "__main__":
#     print("1", state)
#     state["answer"] = "hahaha"
#     print("2", state)
