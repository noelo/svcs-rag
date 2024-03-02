import openai
import logging
from bs4 import BeautifulSoup as Soup
from typing import List
from datetime import datetime
import chainlit as cl
import chainlit.data as cl_data
from customchainlitds import TestDataLayer

from llama_index.callbacks.global_handlers import set_global_handler

from llama_index import (
    StorageContext,
    load_index_from_storage,
)

from llama_index.vector_stores.types import (
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult)
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    get_response_synthesizer,
)
from llama_index.schema import Document
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.retrievers import VectorIndexRetriever
from llama_index.retrievers import BM25Retriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index.prompts import PromptTemplate
from llama_index.callbacks.base import CallbackManager
from llama_index.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings
from tqdm.asyncio import tqdm

from llama_index import QueryBundle
from llama_index.retrievers import BaseRetriever
from typing import Any, List
from llama_index.schema import NodeWithScore


set_global_handler("simple")

logging.basicConfig(level=logging.DEBUG)

db2 = chromadb.PersistentClient(path="/var/home/noelo/dev/svcs-rag/chromadb",settings=Settings(anonymized_telemetry=False))
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store)
docstore = SimpleDocumentStore.from_persist_path("/var/home/noelo/dev/svcs-rag/chromadb/docstore.json")


vector_retriever = VectorIndexRetriever(
    vector_store_query_mode=VectorStoreQueryMode.SEMANTIC_HYBRID,
    index=index,
    similarity_top_k=10,
    verbose=True
)

bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore, 
    similarity_top_k=2,
    verbose=True
)

async def run_queries(queries, retrievers):
    """Run queries against retrievers."""
    tasks = []
    for query in queries:
        for i, retriever in enumerate(retrievers):
            tasks.append(retriever.aretrieve(query))

    task_results = await tqdm.gather(*tasks)

    results_dict = {}
    for i, (query, query_result) in enumerate(zip(queries, task_results)):
        results_dict[(query, i)] = query_result

    return results_dict

def fuse_results(results_dict, similarity_top_k: int = 2):
    """Fuse results."""
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for nodes_with_scores in results_dict.values():
        for rank, node_with_score in enumerate(
            sorted(
                nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True
            )
        ):
            text = node_with_score.node.get_content()
            text_to_node[text] = node_with_score
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: List[NodeWithScore] = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score

    return reranked_nodes[:similarity_top_k]

class FusionRetriever(BaseRetriever):
    """Ensemble retriever with fusion."""

    def __init__(
        self,
        llm,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def retrieve(query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        results_dict = run_queries(query_bundle.query_str, [vector_retriever, bm25_retriever])
        final_results = fuse_results(
            results_dict, similarity_top_k=2
        )
        return final_results


qa_prompt = PromptTemplate(
    """\
Context information is below.
---------------------
{context_str}
---------------------
Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "source" part in your answer.
The "source" part should be a reference to the 'source:' of the document from which you got your answer.
Given the context information and not prior knowledge, answer the query. The "source" part should be one per line.

Example of your response should be:

```
The answer is foo

source: 
    xyz
    abc
    def
```

Query: {query_str}
Answer: \
"""
)

cl_data._data_layer = TestDataLayer()

async def callLLM(query_engine:RetrieverQueryEngine,query:str):
    # results_dict = await run_queries(query, [vector_retriever, bm25_retriever])
    # final_results = fuse_results(results_dict)
    # print(final_results)
    return query_engine.query(query)



@cl.on_chat_start
async def factory():
    service_context = ServiceContext.from_defaults(
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_prompt,
    )

    # assemble query engine
    query_engine = RetrieverQueryEngine.from_args(
        # retriever=vector_retriever,
        # retriever=bm25_retriever,
        retriever=FusionRetriever,
        service_context=service_context,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        response_mode=ResponseMode.ACCUMULATE,
        verbose=True,
        streaming=True,
    )
    cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine") 
    
    # response = await cl.make_async(query_engine.query)(message.content)
    # response = await cl.make_async(callLLM)(query_engine,message.content)
    
    response = await callLLM(query_engine,message.content)

    response_message = cl.Message(content=f"{response}")

    # response_message.content = f"{response}"

    await response_message.send()
