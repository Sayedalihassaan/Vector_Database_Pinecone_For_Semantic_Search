from typing import List, Dict, Literal, Optional
from fastapi import HTTPException
from .PineconeClient import PineconeModel
from .EmbeddingModel import EmbeddingHelper

class VectorController:
    """Controller class for vector database operations."""
    
    def __init__(self):
        """Initialize the vector controller with models and helpers."""
        self.pinecone_model = PineconeModel()
        self.embedding_helper = EmbeddingHelper()
    
    def search_vector_db(self, 
                    query_text: str, 
                    top_k: int, 
                    threshold: Optional[float] = None, 
                    class_type: Optional[Literal['class-a', 'class-b']] = None, 
                    namespace: Literal['HF:all-MiniLM-L6-v2', 'HF:all-MiniLM-L12-v2'] = 'HF:all-MiniLM-L6-v2') -> List[Dict]:
        """Search for similar vectors in the vector database.
        """
        try:
            # Get embeddings of the input query
            query_embedding = self.embedding_helper.encode_text(query_text)
            
            # Prepare filter if class_type is provided
            filter_dict = {'class': class_type} if class_type in ['class-a', 'class-b'] else None
            
            # Search in pinecone
            results = self.pinecone_model.query_index(
                vector=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict,
                namespace=namespace
            )
            
            results = results['matches']
            
            # Filter the output if there is a threshold given
            if threshold:
                similar_records = [
                    {
                        'id': int(record['id']), 
                        'score': float(record['score']), 
                        'class': record['metadata']['class']
                    } 
                    for record in results if float(record['score']) > threshold
                ]
            else:
                similar_records = [
                    {
                        'id': int(record['id']), 
                        'score': float(record['score']), 
                        'class': record['metadata']['class']
                    } 
                    for record in results
                ]
                
            return similar_records
            
        except Exception as e:
            raise HTTPException(status_code=500, detail='Failed to get similar records: ' + str(e))
    
    def insert_to_vector_db(self, 
                        text_id: int, 
                        text: str, 
                        class_type: Literal['class-a', 'class-b'], 
                        namespace: Literal['HF:all-MiniLM-L6-v2', 'HF:all-MiniLM-L12-v2'] = 'HF:all-MiniLM-L6-v2'):
        """Insert a new vector to the vector database.
        """
        try:
            # Get embeddings using the embedding helper
            embedding = self.embedding_helper.encode_text(text)
            
            # Prepare data for pinecone
            to_upsert = [(str(text_id), embedding, {'class': class_type})]
            
            # Insert to pinecone
            self.pinecone_model.upsert_vectors(vectors=to_upsert, namespace=namespace)
            
            # Get the count of vectors after upserting
            count_after = self.pinecone_model.get_index_stats()['namespaces'][namespace]['vector_count']
            
            return {f'Upserting Done in {namespace}: Vectors Count Now is: {count_after} records'}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to upsert to {namespace} vector DB: {str(e)}')
    
    def delete_from_vector_db(self, 
                            text_id: int, 
                            namespace: Literal['HF:all-MiniLM-L6-v2', 'HF:all-MiniLM-L12-v2'] = 'HF:all-MiniLM-L6-v2'):
        """Delete a vector from the vector database.
        """
        try:
            # Delete from vector DB
            self.pinecone_model.delete_vectors(ids=[str(text_id)], namespace=namespace)
            
            # Get the count of vectors after deletion
            count_after = self.pinecone_model.get_index_stats()['namespaces'][namespace]['vector_count']
            
            return {f'Deleting Done in {namespace}: Vectors Count Now is: {count_after} ..'}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to delete from {namespace} vector DB: {str(e)}')