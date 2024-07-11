from text_tools import load_pdf, chunk_text
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from const import env_openai, embedding_model, index_name


if __name__ == "__main__":
    print("Hello langchain")
    text_list = load_pdf("fundasta-docs\\fundasta.pdf")
    chunks = chunk_text(text_list)

    # chunks = [
    #     {"page-content": "aa", "metadata": {"page-number": 1}},
    #     {"page-content": "bb", "metadata": {"page-number": 2}},
    # ]

    embeddings = OpenAIEmbeddings(openai_api_key=env_openai, model=embedding_model)

    text = [x["page-content"] for x in chunks]
    metadata = [x["metadata"] for x in chunks]

    upsert = PineconeVectorStore.from_texts(
        texts=text,
        embedding=embeddings,
        metadatas=metadata,
        index_name=index_name,
    )

    print(PineconeVectorStore.similarity_search("有給休暇", k=3))
