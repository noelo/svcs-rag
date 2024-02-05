import openai
import logging
from bs4 import BeautifulSoup as Soup
from typing import List
from datetime import datetime
import faiss
from llama_index.schema import Document
from llama_index import (
    StorageContext,
    load_index_from_storage,
)
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    get_response_synthesizer,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor

from llama_index.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.prompts import PromptTemplate
from llama_index.callbacks.base import CallbackManager
import chainlit as cl

logging.basicConfig(level=logging.DEBUG)
storage_context = StorageContext.from_defaults(persist_dir="/var/home/noelo/dev/svcs-rag/faissdb")

# load index
index = load_index_from_storage(storage_context)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)



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
Given the context information and not prior knowledge, answer the query.

Example of your response should be:

```
The answer is foo
source: xyz
```

Query: {query_str}
Answer: \
"""
)

refine_prompt = PromptTemplate(
    """\
    The original query is as follows: {query_str}\n
    We have provided an existing answer: {existing_answer}\n
    We have the opportunity to refine the existing answer 
    (only if needed) with some more context below.\n
    ------------\n
    {context_msg}\n
    ------------\n
    Given the new context, refine the original answer to better 
    answer the query. 
    If the context isn't useful, return the original answer.\n
    ALWAYS return a "sources" part in your answer.
    The "sources" part should be a reference to the 'source:' of the document from which you got your answer.
    Given the context information and not prior knowledge, answer the query.

    Example of your response should be:
    ```
    The answer is foo
    sources: \n xyz \n
    ```"
    Refined Answer: 
"""
)

@cl.on_chat_start
async def factory():
    service_context = ServiceContext.from_defaults(callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),)

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(text_qa_template=qa_prompt,refine_template=refine_prompt)

    # assemble query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        service_context=service_context,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        response_mode=ResponseMode.NO_TEXT,
        verbose=True, 
        streaming=True,
    )
    cl.user_session.set("query_engine", query_engine)

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine
    response = await cl.make_async(query_engine.query)(message.content)

    response_message = cl.Message(content="")

    print (response.__class__)
    print (response.__class__.__name__)
    print (response)


    response_message.content = f"{response}"

    await response_message.send()
