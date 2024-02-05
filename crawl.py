import os
import json
import openai
import logging
import requests
from bs4 import BeautifulSoup as Soup
from typing import List
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings


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
                print("Finished...")
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
        page_content=page_content,
        metadata={
            **main_meta,
        },
    )
    return doc
    
def main():
    '''Tokenizes, embds and stores embeddings'''
    openai.api_key = os.environ["OPENAI_API_KEY"]
    root_url = os.environ["ROOT_URL"]

    llm_embedding = OpenAIEmbeddings()

    # TODO: remove
    # root_url = "https://www.redhat.com/rhdc/jsonapi/solr_search/training"

    pages = crawl_site(root_url)
    documents= load_and_parse_documents(pages)
    logging.info(f"Returned {len(documents)}")

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    logging.info(f"RecursiveCharacterTextSplitter Start : {datetime.now()}")
    docs = text_splitter.split_documents(documents)
    logging.info(f"RecursiveCharacterTextSplitter End : {datetime.now()}")
    logging.info(f"RecursiveCharacterTextSplitter Split into {len(docs)} chunks of text")
       

    db = FAISS.from_documents(docs, llm_embedding)
    db.save_local("/var/home/noelo/dev/svcs-rag/faissdb")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
