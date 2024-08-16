from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(text: str) -> np.ndarray:
    return model.encode([text])[0]


def retrieve_relevant_info(query: str, memories: list = None) -> str:
    if not memories:
        # If no memories are provided, you might want to fetch them from your MemoryManager
        # For now, we'll just return an empty string
        return ""

    query_embedding = get_embeddings(query)
    memory_embeddings = [get_embeddings(memory["content"]) for memory in memories]

    similarities = cosine_similarity([query_embedding], memory_embeddings)[0]
    most_relevant_idx = np.argmax(similarities)

    return memories[most_relevant_idx]["content"]
