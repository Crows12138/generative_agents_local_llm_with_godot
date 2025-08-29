import time
import pytest

try:  # Optional heavy dependency
    from basic_functions.perception.embedding import get_embedding
except ModuleNotFoundError:  # pragma: no cover - skip if sentence_transformers unavailable
    pytest.skip(
        "sentence_transformers not available",
        allow_module_level=True,
    )

from basic_functions.memory.memory import Memory

def test_memory_add_and_retrieve():
    mem = Memory()

    texts = ["see a red door", "see a blue box", "meet a green agent"]
    embs = get_embedding(texts)

    for txt, emb in zip(texts, embs):
        mem.add(text=txt, embedding=emb)
    
    top2 = mem.retrieve_similar(query_emb=embs[0], top_k=2)
    assert len(top2) == 2
    assert top2[0].text == texts[0]
    assert top2[1].text == texts[1]

def test_expiry_ttl(monkeypatch):
    mem = Memory()
    emb = [0.1] * 5

    fake_time = [0.0]

    monkeypatch.setattr(time, "time", lambda: fake_time[0])

    mem.add(text="short term memory", embedding=emb, ttl=1.0)
    assert len(mem.entries) == 1

    fake_time[0] += 1.1
    retrieved = mem.retrieve_similar(query_emb=emb)
    assert len(mem.entries) == 0
    assert retrieved == []

def test_filter_event_type():
    mem = Memory()
    emb_good = [0.2] * 5
    emb_bad = [0.3] * 5

    mem.add(text="open door", embedding=emb_good, event_type="action")
    mem.add(text="see enemy", embedding=emb_bad, event_type="combat")

    out = mem.retrieve_similar(query_emb=emb_good, top_k=5, filter_event="action")
    assert len(out) == 1
    assert out[0].text == "open door"

    out2 = mem.retrieve_similar(query_emb=emb_good, top_k=2, filter_event=None)
    assert len(out2) == 2


def test_duplicate_increases_importance(monkeypatch):
    mem = Memory()
    emb = [0.1] * 5

    fake_time = [0.0]
    monkeypatch.setattr(time, "time", lambda: fake_time[0])

    mem.add(text="repeat me", embedding=emb, importance=0.5)
    assert len(mem.entries) == 1
    first = mem.entries[0]
    first_ts = first.timestamp
    first_imp = first.importance
    first_freq = first.metadata["freq"]

    fake_time[0] += 5.0
    mem.add(text="repeat me", embedding=emb, importance=0.5)
    assert len(mem.entries) == 1
    updated = mem.entries[0]
    assert updated.timestamp > first_ts
    assert updated.metadata["freq"] == first_freq + 1
    assert updated.importance > first_imp
