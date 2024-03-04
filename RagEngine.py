import logging
from typing import List
from typing import Any, List

from llama_index.callbacks.global_handlers import set_global_handler
from llama_index.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)

from llama_index.vector_stores.types import VectorStoreQueryMode
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    get_response_synthesizer,
)

from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.retrievers import VectorIndexRetriever
from llama_index.retrievers import BM25Retriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.prompts import PromptTemplate
from llama_index.callbacks.base import CallbackManager
from llama_index.response_synthesizers import ResponseMode
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings


from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from FusionRetriever import FusionRetriever


class RagEngine:
    
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)

        set_global_handler("simple")
        
        logging.info("loading from FlagEmbeddingReranker")
        self.reranker = FlagEmbeddingReranker(
            top_n=10,
            model="BAAI/bge-reranker-large",
            use_fp16=False
        )

        logging.info("loading from ChromaDB")
        db2 = chromadb.PersistentClient(path="/var/home/noelo/dev/svcs-rag/chromadb",settings=Settings(anonymized_telemetry=False))
        chroma_collection = db2.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        index = VectorStoreIndex.from_vector_store(vector_store)
        docstore = SimpleDocumentStore.from_persist_path("/var/home/noelo/dev/svcs-rag/chromadb/docstore.json")
        logging.info("loading from ChromaDB....done")


        logging.info("loading from FlagEmbeddingReranker....done")


        self.vector_retriever = VectorIndexRetriever(
            vector_store_query_mode=VectorStoreQueryMode.SEMANTIC_HYBRID,
            index=index,
            similarity_top_k=10,
            verbose=True
        )

        self.bm25_retriever = BM25Retriever.from_defaults(
            docstore=docstore, 
            similarity_top_k=20,
            verbose=True
        )


        qa_prompt = PromptTemplate(
            """\
        Context information is below, it is a set of documents containing the title and summary of training courses, exams and solutions provided by Red Hat that are relevant to the users question.
        ---------------------
        {context_str}
        ---------------------
        Summerize and combine each document contained in the Context information to build a response to the user. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        ALWAYS return a "source" part in your answer. If you don't know the answer leave the "source" part empty.
        The "source" part should be a reference to the 'source:' of the documents provided in the Context Information. 

        Example of your response should be:

        ```
        The answer is foo

        source: 
            1. xyz
            2. abc
            3. def
        ```

        Query: {query_str}
        Answer: \
        """
        )
        
        service_context = ServiceContext.from_defaults(
            callback_manager=CallbackManager([LlamaDebugHandler()]),
        )

        # configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            service_context=service_context,
            text_qa_template=qa_prompt,
            verbose=True
        )

        self.FusionRetriever = FusionRetriever(retrievers=[self.vector_retriever, self.bm25_retriever])

        # assemble query engine
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.FusionRetriever,
            service_context=service_context,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[self.reranker],
            response_mode=ResponseMode.TREE_SUMMARIZE,
            verbose=True,
            streaming=True,
        )
        
    def query(self,query:str):
        resp =  self.query_engine.query(query)
        return resp
        

