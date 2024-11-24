import os
import argparse
from tqdm import tqdm
from pprint import pprint
from dotenv import load_dotenv

import numpy as np
import pymupdf4llm
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src import embed_utils

load_dotenv()
Questions = [
    "What is the name of the company?",
    "Who is the CEO of the company?",
    "What is their vacation policy?",
    "What is the termination policy?",
]

parser = argparse.ArgumentParser(description="RAG system - stateless")

parser.add_argument(
    "--pdf_path", type=str, default="source_docs/handbook.pdf", help="example : ./a.pdf"
)
parser.add_argument(
    "--questions", nargs="*", type=str, default=Questions, help="List of questions"
)


args = parser.parse_args()
openai_key = os.getenv("openai_key")

embedding_model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-small-en",  # switch to en/zh for English or Chinese
    trust_remote_code=True,
)
llm_model = OpenAI(api_key=openai_key)
embedding_model.max_seq_length = 2048


print("PDF Parsing and Vectorizing ...")
pdf_path_, embedding_vector_, text_chunks_ = embed_utils.create_embeddings(
    model=embedding_model, pdf_path=args.pdf_path, batch_size=2
)

response_list = []
print("LLM Execution...")
for quest in args.questions:
    query_, context_ = embed_utils.get_context(
        model=embedding_model,
        prompt=quest,
        embedding_vector=embedding_vector_,
        text_chunks=text_chunks_,
        confidence_threshold=0.7,
    )

    response = embed_utils.llm_agent(model=llm_model, query=query_, context=context_)

    pprint(response)
    response_list.append(response)
