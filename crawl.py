import os
import json
import openai
import logging
import requests
from bs4 import BeautifulSoup as Soup
from typing import List
from datetime import datetime

from llama_index.schema import Document
from llama_index import (
    VectorStoreIndex,
    StorageContext,
)
# import faiss
# from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from llama_index import VectorStoreIndex
from llama_index.embeddings import OpenAIEmbedding
from llama_index.text_splitter import SentenceSplitter
from llama_index.ingestion import IngestionPipeline, IngestionCache
from llama_index import ServiceContext


def crawl_site(root_url:str) -> {}:
    ''' Crawl website'''
    page_list = []
    try:
        for i in range(20):
            print(f"Page {i}")
            page = requests.get(
                root_url,
                params={"page": i, "language": "en"},
            )
            json_dict = json.loads(page.text)

            if (len(json_dict["body"]["docs"])) < 1:
                print("Finished crawling...")
                break
            for i in json_dict["body"]["docs"]:
                page_list.append(
                    {"description": i["description"], "url": i["url"], "title": i["title"]}
                )
        return page_list
    except requests.exceptions.RequestException as e:  # Ignore
        raise SystemExit(e)
    
def load_and_parse_documents(document_dict:{}) -> List[Document]:
    '''scrape the documents and parse the contents'''
    return_docs=[]
    for single_document in document_dict:
        page = requests.get(single_document['url'])
        soup = Soup(page.content,'html.parser')
        page_text = soup.find_all(attrs={"class": "rh-generic--component"})

        page_content=''	
        for item in page_text:   
            page_content = page_content+" ".join(item.text.strip().split())

        return_docs.append(page_to_document(single_document,page_content))
    return return_docs


def page_to_document(page_meta_data: {}, page_content: str) -> Document:
    ''' Build the LC document'''
    main_meta = {
        "title": page_meta_data['title'],
        "summary": page_meta_data['description'],
        "source": page_meta_data['url'],
    }

    doc = Document(
        text=page_content,
        metadata={**main_meta,},
    )
    return doc
    
def main():
    '''Tokenizes, embds and stores embeddings'''
    openai.api_key = os.environ["OPENAI_API_KEY"]
    root_url = os.environ["ROOT_URL"]

    pages = crawl_site(root_url)
    documents= load_and_parse_documents(pages)
    logging.info(f"Returned {len(documents)} Documents")

    db = chromadb.PersistentClient(path="/var/home/noelo/dev/svcs-rag/chromadb")
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=200, chunk_overlap=50),
            OpenAIEmbedding(),
        ],
        vector_store=vector_store
    )

    nodes = pipeline.run(documents=documents)
    
    index = VectorStoreIndex(nodes=nodes,show_progress=True)
    print(f"Nodes created {len(nodes)}")
    index.storage_context.persist(persist_dir="/var/home/noelo/dev/svcs-rag/chromadb")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
