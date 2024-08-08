#
# Source: https://github.com/rh-aiservices-bu/llm-on-openshift/blob/main/examples/notebooks/langchain/Langchain-Milvus-Ingest-nomic.ipynb
#

# Needed packages and imports
# pip install -r requirements.txt

import os
import requests
import urllib.request
import json

from dotenv import load_dotenv
from pathlib import Path/home/stefan/Projects/langchain/carambola

from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

doc_list_local = False

if doc_list_local:
    f = open("data/doc-list.json")
    doc_list = json.load(f)
else:
    DOCLIST = os.getenv("DOCLIST")
    with urllib.request.urlopen(DOCLIST) as url:
        doc_list = json.load(url)

# Base parameters, the Milvus connection info

# MILVUS_HOST = "vectordb-milvus.milvus.svc.cluster.local"
# oc port-forward  service/vectordb-milvus 19530:19530 -n milvus
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_USERNAME = os.getenv("MILVUS_USERNAME")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
MILVUS_COLLECTION = "collection_nomicai_embeddings"


# Initial index creation and document ingestion
/home/stefan/Projects/langchain/carambola
# Download and load pdfs

pdfs = doc_list["pdfs"]

pdfs_to_urls = {}
for url in pdfs:
    pdfs_to_urls[Path(url).name] = url


docs_dir = f"docs"

if not os.path.exists(docs_dir):
    os.mkdir(docs_dir)

for pdf in pdfs:
    try:
        response = requests.get(pdf)
    except:
        print(f"Skipped {pdf}")
        continue
    if response.status_code != 200:
        print(f"Skipped {pdf}")
        continue
    with open(f"{docs_dir}/{pdf.split('/')[-1]}", "wb") as f:
        f.write(response.content)


pdf_loader = PyPDFDirectoryLoader(docs_dir)
pdf_docs = pdf_loader.load()

# Inject metadata

for doc in pdf_docs:
    doc.metadata["source"] = pdfs_to_urls[Path(doc.metadata["source"]).name]

# Load websites

websites = doc_list["websites"]
website_loader = WebBaseLoader(websites)
website_docs = website_loader.load()

# Merge both types of docsfrom langchain_community.vectorstores import Milvus
docs = pdf_docs + website_docs

# Split documents into chunks with some overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=40)
all_splits = text_splitter.split_documents(docs)
all_splits[0]


# Create the index and ingest the documents

# If you don't want to use a GPU, you can remove the 'device': 'cuda' argument
# model_kwargs = {'trust_remote_code': True, 'device': 'cuda'}
/home/stefan/Projects/langchain/carambola
model_kwargs = {"trust_remote_code": True}

embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs=model_kwargs,
    show_progress=True,
)


db = Milvus(
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
    drop_old=True,
)

db.add_documents(all_splits)


query = "What is a OpenShift subscription?"
query = "Each self-managed Red Hat OpenShift subscription includes entitlements for Red Hat OpenShift, Red Hat Enterprise Linux, and other OpenShift-related components. These entitlements are included for running OpenShift control plane and infrastructure workloads and do not need to be accounted for when determining the number of subscriptions needed."

docs_with_score = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
