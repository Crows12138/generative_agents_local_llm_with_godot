from typing import List

try:
    from sentence_transformers import SentenceTransformer

    MODEL = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)
except Exception:  # pragma: no cover - fallback when model is unavailable
    import string

    class _DummyModel:
        """Fallback embedding model returning simple character-frequency vectors."""

        _ALPHABET = string.ascii_lowercase

        def encode(self, texts: List[str]):
            if isinstance(texts, str):
                texts = [texts]
            vectors = []
            for text in texts:
                lower = text.lower()
                vec = [lower.count(ch) for ch in self._ALPHABET]
                vectors.append(vec)
            return vectors

    MODEL = _DummyModel()

def get_embedding(texts: List[str]) -> List[List[float]]:
    embs = MODEL.encode(texts)
    result = []
    for emb in embs:
        if hasattr(emb, "tolist"):
            result.append(emb.tolist())
        else:
            result.append(list(emb))
    return result

def get_single_embedding(text: str) -> List[float]:
    return get_embedding([text])[0]
