from pinecone import Pinecone
from src.config import PINECONE_API_KEY

class PineconeModel:
    """Model class for Pinecone vector database operations."""
    
    def __init__(self, index_name: str = 'quickstart'):
        """Initialize the Pinecone connection.
        """
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(name=index_name)
    
    def query_index(self, 
                vector, 
                top_k: int, 
                namespace: str,
                filter_dict = None, 
                include_metadata: bool = True):
        """Query the Pinecone index.
        """
        if filter_dict:
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                filter=filter_dict,
                namespace=namespace,
                include_metadata=include_metadata
            )
        else:
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=include_metadata
            )
        return results
    
    def upsert_vectors(self, vectors, namespace: str):
        """Upsert vectors to the Pinecone index.
        """
        response = self.index.upsert(
            vectors=vectors,
            namespace=namespace
        )
        return response
    
    def delete_vectors(self, ids, namespace: str):
        """Delete vectors from the Pinecone index.
        """
        response = self.index.delete(
            ids=ids,
            namespace=namespace
        )
        return response
    
    def get_index_stats(self):
        """Get statistics about the Pinecone index.
        """
        return self.index.describe_index_stats()