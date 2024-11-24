import os
import argparse
import pandas as pd
from tqdm import tqdm
from glob import glob
from typing import List,Dict
from pprint import pprint

import numpy as np
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer

import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter



headers_to_split_on = [
	("#", "Header 1"),
	("##", "Header 2"),
	("###", "Header 3"),
	("####", "Header 4")]



def create_embeddings(model,pdf_path:str,batch_size:int) -> (str,np.ndarray,List[str]):
	"""
	Reads the PDF file, chunks it and vectorizes the chunks.
	finally creates a dataframe
		Args: 
			pdf_path - PDF File path
			batch_size - Model batch size to run inference
		Returns:
			 Embedding vector and text chunks
	"""
	md_text = pymupdf4llm.to_markdown(pdf_path,show_progress=True)
	markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
	md_header_splits  = markdown_splitter.split_text(md_text)
	
	text_chunks = []
	for i,elements in enumerate(md_header_splits):
		context_setter = "The topics discussed is here about : "+", ".join(list(md_header_splits[i].metadata.values()))
		context_phrase = str(md_header_splits[i].page_content)
		chunk_ = context_setter+"\n"+context_phrase
		text_chunks.append(chunk_)
	
	embeddings = model.encode(text_chunks,batch_size=1)
	return pdf_path,embeddings,text_chunks


def get_context(model,prompt:str,embedding_vector:np.ndarray,text_chunks:List,confidence_threshold:float=0.8)->str:
    """
    Retrives the context passage from the document corpus
    using cosine similarity.

    Args:
        prompt - User prompt.
        embedding_vector - Embeddings to search.
        confidence_threshold - confidence threshold for filtering documents.
    Return:
        Context paragraph or Passage
    """
    context_passage = ''
    query_vector = model.encode([prompt])
    scores = cos_sim(query_vector,embedding_vector)[0]
    max_index = np.argmax(scores).item()
    
    if scores[max_index]>=confidence_threshold:
        context_passage+=text_chunks[max_index]    
    context_passage+="None"
    return prompt,context_passage


def llm_agent(model,query:str,context:str)->Dict:
    """
    A Simple llm agent to answer user query given context.
    Args:
        model : LLM model
        query : User query
        context :  Passage to fetch answer.

    Returns:
        A Dict with question and its answer.
    """
    base_context = f"From the given company handbook answer the user query accurately, if not confident respond None. {context}"
    completion = model.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": base_context+f" Question: {query}"
            }
        ],max_tokens=512,temperature=0.0,
    )

    llm_response = completion.choices[0].message.content
    return {query:llm_response}
