# generative_agents/perception/perceive.py
from typing import Dict, Any, List
from .describe import gather_surrounding_descriptions
from .embedding import get_embedding

def perceive(persona, maze, radius: int = 5) -> Dict[str, Any]:
    descriptions: List[str] = gather_surrounding_descriptions(persona, maze, radius)
    if not descriptions:
        return {
            "descriptions": [],
            "embeddings": [],
            "merged_embedding": []
        }
    embeddings: List[List[float]] = get_embedding(descriptions)
    dim = len(embeddings[0])
    merged = [0.0] * dim
    for emb in embeddings:
        for i, v in enumerate(emb):
            merged[i] += v
    merged_embedding = [v / len(embeddings) for v in merged]
    return {
        "descriptions": descriptions,
        "embeddings": embeddings,
        "merged_embedding": merged_embedding
    }
