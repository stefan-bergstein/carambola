# Needed packages and imports

import os
from dotenv import load_dotenv

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import VLLMOpenAI
from langchain.prompts import PromptTemplate
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

# Bases parameters, Inference server and Milvus info

load_dotenv()


# INFERENCE_SERVER_URL = "http://vllm.llm-hosting.svc.cluster.local:8000/v1"
INFERENCE_SERVER_URL = "http://localhost:8000/v1"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_TOKENS = 1024
TOP_P = 0.95
TEMPERATURE = 0.01
PRESENCE_PENALTY = 1.03

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_USERNAME = os.getenv("MILVUS_USERNAME")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
MILVUS_COLLECTION = "collection_nomicai_embeddings"

# Initialize the connection

model_kwargs = {"trust_remote_code": True}
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs=model_kwargs,
    show_progress=False,
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

# Initialize query chain

template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant named HatBot answering questions.
You will be given a question you need to answer, and a context to provide you with information. You must answer the question based as much as possible on this context.
Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Context: 
{context}

Question: {question} [/INST]
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base=INFERENCE_SERVER_URL,
    model_name=MODEL_NAME,
    max_tokens=MAX_TOKENS,
    top_p=TOP_P,
    temperature=TEMPERATURE,
    presence_penalty=PRESENCE_PENALTY,
    streaming=True,
    verbose=False,
    callbacks=[StreamingStdOutCallbackHandler()],
)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=store.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Query example

question = "What is a OpenShift subscription?"
result = qa_chain.invoke({"query": question})


# Retrieve source
def remove_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item.metadata["source"] not in unique_list:
            unique_list.append(item.metadata["source"])
    return unique_list


results = remove_duplicates(result["source_documents"])

for s in results:
    print(f"{s}")
