import os
from data.const import GraphState, env_openai, embedding_model, index_name, env_pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import traceback
from module.tools.utils import format_docs

os.environ["OPENAI_API_KEY"] = env_openai
os.environ["PINECONE_API_KEY"] = env_pinecone


#   RAG document retrieval
def retrieve_document(state: GraphState) -> GraphState:

    try:

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
