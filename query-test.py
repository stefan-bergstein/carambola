import os
from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_USERNAME = os.getenv("MILVUS_USERNAME")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
MILVUS_COLLECTION = "collection_nomicai_embeddings"


model_kwargs = {"trust_remote_code": True}

embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs=model_kwargs,
    show_progress=True,
)

store = Milvus(
    embedding_function=embeddings,
    connection_args={
        "host": MILVUS_HOST,
        "port": MILVUS_PORT,
        "user": MILVUS_USERNAME,
        "password": MILVUS_PASSWORD,
    },
    collection_name=MILVUS_COLLECTION,
    enable_dynamic_field=True,
    text_field="page_content",
    auto_id=True,
    drop_old=False,
)

# Make a query to the index to verify sources

query = "What is a OpenShift subscription?"
results = store.similarity_search_with_score(query, k=4, return_metadata=True)
for result in results:
    print(result[0].metadata["source"])

# Work with a retriever

retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

docs = retriever.invoke(query)
