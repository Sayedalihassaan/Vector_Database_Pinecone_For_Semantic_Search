from sentence_transformers import SentenceTransformer

class EmbeddingHelper:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name_or_path=model_name, device=device)

    def encode_text(self, text: str) -> list:
        return self.model.encode(text).tolist()