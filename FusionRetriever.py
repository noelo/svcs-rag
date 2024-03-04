import logging
from typing import List
from llama_index import QueryBundle
from llama_index.retrievers import BaseRetriever
from llama_index.schema import NodeWithScore

class FusionRetriever(BaseRetriever):
    """Ensemble retriever with fusion."""

    def __init__(
        self,
        retrievers: List[BaseRetriever]
    ) -> None:
        """Init params."""
        self.retrievers = retrievers
        super().__init__()        
        

    def _retrieve(self,query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        logging.info("Performing retrieve")
        node_list = self.run_queries(query_bundle.query_str)
        return node_list
    
    def run_queries(self,query):
        """Run queries against retrievers."""

        results_list= []
        for i, retriever in enumerate(self.retrievers):
                query_result = retriever.retrieve(query)
                for value in query_result:
                    results_list.append(value)
                    logging.debug(f"{value}")
        logging.info(f"run_queries returning {len(results_list)} results")
        return results_list