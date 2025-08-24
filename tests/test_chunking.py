from corpusloom.chunking import Chunker

def test_chunker_basic():
    ch = Chunker(max_tokens=50, overlap_tokens=10)
    text = "## Header\n\nThis is a paragraph.\n\n More Text for Test." * 80
    chunks = ch.chunk_text(text)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c for c in chunks)


def test_chunker_empty_returns_empty_list():
    assert Chunker().chunk_text("") == []

def test_chunker_windows_newlines_and_code_block():
    text = "A\r\n\r\n```py\r\nprint(1)\r\n```\r\n\r\nB"
    out = Chunker().chunk_text(text)
    # Expect at least 3 blocks: "A", code fence, "B" (order may vary around pre/code split)
    assert any(b.strip() == "A" for b in out)
    assert any(b.strip().startswith("```") and b.strip().endswith("```") for b in out)
    assert any(b.strip() == "B" for b in out)

def test_chunker_forces_multiple_chunks_when_small_max():
    c = Chunker(max_tokens=8, overlap_tokens=2)
    long_text = "x" * 200  # ~50 tokens with the default heuristic -> multiple chunks
    out = c.chunk_text(long_text)
    assert len(out) >= 2

def test_chunker_small_paragraphs_fastpath_multiple_blocks():
    text = "Alpha\n\nBeta\n\nGamma"
    out = Chunker(max_tokens=800).chunk_text(text)
    # Fast-path should keep separate blocks when they fit within one chunk
    assert out == ["Alpha", "Beta", "Gamma"]
