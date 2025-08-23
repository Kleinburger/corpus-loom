from corpusloom.chunking import Chunker

def test_chunker_basic():
    ch = Chunker(max_tokens=50, overlap_tokens=10)
    text = "## Header\n\nThis is a paragraph. " * 20
    chunks = ch.chunk_text(text)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c for c in chunks)
