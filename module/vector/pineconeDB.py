import os
from data.const import GraphState, env_openai, embedding_model, index_name, env_pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import traceback
from module.tools.utils import format_docs
from langchain.tools.base import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import Tool


os.environ["OPENAI_API_KEY"] = env_openai
os.environ["PINECONE_API_KEY"] = env_pinecone

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    embeddings = OpenAIEmbeddings(openai_api_key=env_openai, model=embedding_model)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 6}
    )
    print("\n\n!!!!!Pinecone initialized successfully.!!!!!\n\n")
except Exception as e:
    print(f"\n\nError initializing Pinecone: {str(e)}, key : {env_pinecone[:5]}")
    print(f"\n\nError initializing Pinecone: {str(e)}")
    print(f"Error type: {type(e)}")
    print(f"Pinecone API key (first 5 chars): {env_pinecone[:5]}")
    print(f"Index name: {index_name}")
    print(f"Traceback: {traceback.format_exc()}")


#   RAG document retrieval
def retrieve_document(state: GraphState) -> GraphState:

    # Retrieves related refence from VectorDB
    retrieved_docs = retriever.invoke(state["question"])
    # Reshape the data
    retrieved_docs = format_docs(retrieved_docs)

    # Preserve it in a GraphState
    return GraphState(
        question=state["question"],
        answer=state["answer"],
        context=retrieved_docs,
        chat_history=state["chat_history"],
        web=state["web"],
        relevance=state["relevance"],
    )


def retrieve_document_str(str):

    # Retrieves related refence from VectorDB
    retrieved_docs = retriever.invoke(state["question"])
    # Reshape the data
    retrieved_docs = format_docs(retrieved_docs)

    # Preserve it in a GraphState
    return retrieved_docs


class FundastA_Policy(BaseTool):
    name: str = "FundastA_Policy"
    description: str = (
        """FundastAの就業規則の内容が確認できるツールです。
    ユーザーがFundastAについて質問している場合、特に就業規則について
    質問しているときに、関連する内容を捜すことが出来ます。関連する内容が
    あった場合にはそれを使ってユーザーの質問に答えてください"""
    )

    def _run(self, input_str: str) -> str:

        # Retrieves related refence from VectorDB
        retrieved_docs = retriever.invoke(input_str)
        # Reshape the data
        retrieved_docs = format_docs(retrieved_docs)

        return retrieved_docs + "\n"

    def _arun(self, input_str: str):
        raise NotImplementedError("Async method not implemented")


# Example usage of the function and class
if __name__ == "__main__":
    # Example GraphState
    state = GraphState(
        question="有給休暇",
        answer="",
        context="",
        chat_history=[],
        web="",
        relevance=0.0,
    )

    # Using the function
    new_state = retrieve_document(state)
    print("Function :", new_state)

    # Using the class
    tool = FundastA_Policy()
    result = tool._run("有給休暇")
    print("\n\nClass :", result)
