import os
import sys
import types
import importlib
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def conditional_stub_embedding(monkeypatch):
    # If we’re not in a CI environment (CI env var not set), do nothing
    if not os.getenv("CI"):
        yield
        return

    # The following only runs when CI=true

    class DummyModel:
        def encode(self, texts):
            # Produce a fake embedding: a vector of length‐5 where each element is the text length
            return np.array([[float(len(t))] * 5 for t in texts])

    # Stub out the sentence-transformers import so it won’t try to download models
    if "sentence_transformers" not in sys.modules:
        stub = types.ModuleType("sentence_transformers")
        setattr(stub, "SentenceTransformer", lambda *args, **kwargs: DummyModel())
        sys.modules["sentence_transformers"] = stub

    # Monkey‐patch our own embedding module’s functions
    embedding_mod = importlib.import_module("basic_functions.perception.embedding")

    def fake_get_embedding(texts):
        # Return the same fake vectors as DummyModel.encode
        return [[float(len(t))] * 5 for t in texts]

    def fake_get_single_embedding(text):
        return fake_get_embedding([text])[0]

    monkeypatch.setattr(embedding_mod, "get_embedding", fake_get_embedding, raising=False)
    monkeypatch.setattr(embedding_mod, "get_single_embedding", fake_get_single_embedding, raising=False)

    # Also stub any direct imports in the test file itself
    test_mod = importlib.import_module("basic_functions.tests.test_perception")
    monkeypatch.setattr(test_mod, "get_embedding", fake_get_embedding, raising=False)
    monkeypatch.setattr(test_mod, "get_single_embedding", fake_get_single_embedding, raising=False)

    yield