import logging
import chainlit as cl
import chainlit.data as cl_data
from customchainlitds import TestDataLayer
from  RagEngine import RagEngine
from llama_index.query_engine import RetrieverQueryEngine
import debugpy


logging.basicConfig(level=logging.DEBUG)
debugpy.listen(('0.0.0.0', 5678))



cl_data._data_layer = TestDataLayer()
query_engine = RagEngine()

def callLLM(query_engine:RetrieverQueryEngine,query:str):
    return query_engine.query(query)

@cl.on_chat_start
async def factory():
    cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")         
    response = await cl.make_async(callLLM)(query_engine,message.content)
    response_message = cl.Message(content=f"{response}")
    await response_message.send()
