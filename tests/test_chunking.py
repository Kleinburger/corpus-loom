from corpusloom.chunking import (
    Chunker,
    _hard_wrap,
    _split_long_block,
    _split_paragraphs,
)


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


def test_chunker_code_only_block_no_pre_no_tail():
    # Covers branches where pre.strip() is False and tail.strip() is False.
    text = "```py\nprint(1)\n```"
    out = Chunker().chunk_text(text)
    assert out == [text]


def test_chunker_overlap_disabled_else_branch():
    # Force the else-branch (cur_toks + btoks > max and cur is not empty)
    # while overlap is DISABLED (ov_chars == 0).
    max_tok = 20
    c = Chunker(max_tokens=max_tok, overlap_tokens=0)

    # First block ~18 tokens, second ~10 tokens -> 18 + 10 > 20 => else path.
    a = "a" * 70  # ceil(70/4) = 18
    b = "b" * 40  # ceil(40/4) = 10
    out = c.chunk_text(a + "\n\n" + b)

    # We should get two chunks. With overlap 0, the second chunk must NOT start
    # with tail of previous chunk (no 'a' overlap).
    assert len(out) >= 2
    assert out[0].startswith("a")
    assert out[1].split("\n", 1)[0].startswith("b")  # first line of second chunk starts with 'b'


def test_chunker_overlap_enabled_else_branch_with_overlap_applied():
    # Same overflow condition, but overlap ENABLED -> second chunk begins with tail of previous.
    max_tok = 20
    c = Chunker(max_tokens=max_tok, overlap_tokens=2)  # ov_chars = 8
    a = "a" * 70  # 18 tokens
    b = "b" * 40  # 10 tokens
    out = c.chunk_text(a + "\n\n" + b)

    assert len(out) >= 2
    # The second chunk should begin with 8 'a' chars (overlap tail), then newline, then b's.
    # Because the chunker inserts overlap then the next block with '\n'.join(cur).
    second = out[1]
    assert second.startswith("a" * 8)


def test_chunker_else_branch_hard_wrap_true_path():
    # Drive the else-branch + subdivision path; piece sizes come from the earlier pre-split
    max_tok = 10
    ov_tok = 2
    c = Chunker(max_tokens=max_tok, overlap_tokens=ov_tok)

    a = "x" * 35  # ceil(35/4) = 9 tokens
    b = "y" * 80  # ceil(80/4) = 20 tokens -> pre-split into 40+40 char pieces
    out = c.chunk_text(a + "\n\n" + b)

    assert len(out) >= 2

    # After joining with "\n", "pure y" chunks may still contain '\n'.
    # Count chunks whose characters are only {'y','\n'} and contain at least one 'y'.
    y_like = [seg for seg in out if seg and set(seg) <= {"y", "\n"} and "y" in seg]
    assert len(y_like) >= 2

    # At least one wrapped piece should be "large enough" (≈ width window is 40 chars)
    assert any(len(seg.replace("\n", "")) >= 35 for seg in y_like)


# Cover the else-branch *non* hard-wrap sub-path (approx_tokens(b) <= 1.25*max)
def test_chunker_else_branch_not_hardwrap_path():
    c = Chunker(max_tokens=10, overlap_tokens=2)
    a = "x" * 35  # 9 tokens
    b = "y" * 36  # 9 tokens (<= 1.25*10) -> else-branch but NOT hard-wrap
    out = c.chunk_text(a + "\n\n" + b)

    assert len(out) >= 2
    assert any("x" in seg for seg in out)
    assert any("y" in seg for seg in out)


def test_chunker_flush_on_exact_max_inside_if_path():
    # Cover the inner "if cur_toks >= self.max_tokens: flush()" path.
    # One block that exactly equals max tokens should cause a flush on append.
    max_tok = 5
    c = Chunker(max_tokens=max_tok, overlap_tokens=0)
    exact = "Z" * (max_tok * 4)  # 20 chars -> 5 tokens exactly
    out = c.chunk_text(exact)
    # Should still produce a single chunk, but the exact-max flush path is taken.
    assert out == [exact]


def test_chunker_if_branch_flush_on_exact_max():
    # overlap 0 to simplify expectations
    c = Chunker(max_tokens=10, overlap_tokens=0)

    a = "a" * 24  # 6 tokens
    b = "b" * 16  # 4 tokens -> 6+4=10 triggers flush within the IF branch
    d = "d" * 24  # 6 tokens -> next chunk

    out = c.chunk_text(a + "\n\n" + b + "\n\n" + d)
    assert len(out) >= 2
    # First chunk contains a+b; second contains d
    assert "a" * 24 in out[0] and "b" * 16 in out[0]
    assert any("d" * 24 in seg for seg in out[1:])


def test_chunker_else_branch_hard_wrap_on_long_code_block():
    c = Chunker(max_tokens=10, overlap_tokens=2)

    pre = "p" * 35  # 9 tokens keeps cur non-empty
    # Long fenced code block -> not split by the "subdivide any single block" stage
    code = "```txt\n" + ("z" * 100) + "\n```"  # ~28 tokens
    text = pre + "\n\n" + code

    out = c.chunk_text(text)
    assert len(out) >= 2
    # Make sure we really processed the fence (pieces will still include backticks)
    assert any("```" in seg for seg in out)


def test_split_long_block_early_return_when_small():
    # Calls helper directly to hit "if n <= char_cap: return [block]".
    block = "short"
    res = _split_long_block(block, max_tokens=100, overlap_tokens=10)
    assert res == [block]


def test_split_paragraphs_heading_and_blank():
    # Exercise heading detection and blank-line flushing.
    text = "Title line\nparagraph line\n\n# Heading 1\nbody"
    blocks = _split_paragraphs(text)
    assert "Title line\nparagraph line" in blocks
    assert "# Heading 1\nbody" in blocks


def test_hard_wrap_prefers_newline_cut_and_filters_empty():
    # Newline inside the window; the middle blank/space-only line is filtered by final list-comp
    s = "part1\n  \npart2"
    pieces = _hard_wrap(s, width_chars=6, overlap_chars=2)
    assert pieces[0] == "part1"
    # No empty pieces after strip-filter
    assert all(p.strip() for p in pieces)
    # Overlap can trim a leading char; allow "part2" or "art2"
    assert any(p.endswith("part2") or p.endswith("art2") for p in pieces)


def test_hard_wrap_prefers_space_cut():
    # No newline within width, but a space exists -> should cut at space.
    s = "aaaa bbbb cccc"
    # width 9 sees "aaaa bbbb" window; prefers space cut (before fallback).
    pieces = _hard_wrap(s, width_chars=9, overlap_chars=2)
    # First piece should include the space cut rather than forcing the full width.
    assert pieces[0].startswith("aaaa")
    assert " " in pieces[0] or pieces[0].endswith("aaaa") is False


def test_hard_wrap_fallback_cut_no_space_no_newline():
    # No spaces or newlines -> fallback to j (hard cut), with overlap stepping.
    s = "q" * 85
    pieces = _hard_wrap(s, width_chars=40, overlap_chars=4)
    # Expect at least 3 pieces due to overlap stepping: 0..40, 36..76, 72..85
    assert len(pieces) >= 3
    assert all(p for p in pieces)  # no empty segments after strip()


def test_hard_wrap_space_cut_without_newline():
    s = "word1 word2 word3"
    # width=12 => threshold is 7.2; last space before j (index 11) is > 7.2 -> use space cut
    pieces = _hard_wrap(s, width_chars=12, overlap_chars=2)
    # First piece should end at the space: "word1 word2"
    assert pieces[0].endswith("word1 word2")
    # No piece should be empty
    assert all(p.strip() for p in pieces)


def test_subdivide_long_non_fenced_block_true_path():
    c = Chunker(max_tokens=10, overlap_tokens=2)
    text = "Z" * 100  # definitely longer than max -> subdivide via _split_long_block
    out = c.chunk_text(text)
    assert len(out) >= 2
    assert all(isinstance(x, str) for x in out)


def test_subdivide_short_non_fenced_block_false_path():
    c = Chunker(max_tokens=50, overlap_tokens=2)
    text = "short paragraph"
    out = c.chunk_text(text)
    assert out == ["short paragraph"]


def test_prepass_long_code_block_is_not_subdivided_here():
    c = Chunker(max_tokens=10, overlap_tokens=2)
    code = "```txt\n" + ("z" * 100) + "\n```"
    text = "p" * 8 + "\n\n" + code
    out = c.chunk_text(text)
    # We still end up with multiple chunks overall, but the fenced block wasn't split in the prepass
    assert len(out) >= 2
    assert any("```" in seg for seg in out)


def test_else_branch_zero_overlap_skips_overlap_copy():
    c = Chunker(max_tokens=10, overlap_tokens=0)
    a = "a" * 35  # ~9 tokens
    b = "b" * 36  # ~9 tokens -> else-branch, NOT hard-wrap
    out = c.chunk_text(a + "\n\n" + b)
    assert len(out) >= 2  # no overlap copied into the next chunk


def test_hard_wrap_fallback_cut_no_space_no_newline():
    s = "abcdefghijk"  # no spaces/newlines -> fallback to cut=j branch
    pieces = _hard_wrap(s, width_chars=5, overlap_chars=1)
    assert len(pieces) >= 2
    assert all(len(p) <= 5 for p in pieces)
    # All characters must come from the original string
    assert all(set(p) <= set(s) for p in pieces)


def test_hard_wrap_add_piece_no_flush_when_fits():
    # Make cur small and the first wrapped piece small enough to avoid the flush condition
    c = Chunker(max_tokens=10, overlap_tokens=2)
    pre = "p" * 4  # ≈ 1 token
    # Use 29 z's so first wrapped piece is 36 chars total:
    # len("```txt\n") == 7, 7 + 29 = 36 -> ceil(36/4) == 9 tokens
    # 1 (overlap) + 9 == 10 -> fits without pre-emptive flush
    code = "```txt\n" + ("z" * 29) + "\n" + ("z" * 200) + "\n```"
    text = pre + "\n\n" + code
    out = c.chunk_text(text)
    assert len(out) >= 2
    # There must be a chunk that contains BOTH the overlap "p" and the first wrapped "z" piece
    # proving that we didn’t flush before appending that first piece.
    assert any(("p" in seg and "z" in seg) for seg in out), out


def test_or_short_circuit_first_block_long_code_not_split():
    """
    Cover the `or not cur` short-circuit where left side is False but right side True.
    We use a fenced code block (not subdivided) whose tokens > max on the very first append.
    """
    max_tok = 10
    c = Chunker(max_tokens=max_tok, overlap_tokens=0)
    code = "```txt\n" + ("w" * 100) + "\n```"  # ~25 tokens; code fences are not split
    out = c.chunk_text(code)
    # Stays as a single chunk; exercised path: (cur_toks + btoks <= max) == False, (not cur) == True
    assert out == [code]


def test_overlap_tail_added_when_nonempty():
    """
    Cover the overlap addition path: `if chunks and ov_chars > 0:` and `if tail_overlap.strip():`
    (true branch). Ensures tail overlap is actually prefixed to the next chunk.
    """
    max_tok = 10
    ov_tok = 2  # ov_chars = 8
    c = Chunker(max_tokens=max_tok, overlap_tokens=ov_tok)
    a = "a" * 32  # 8 tokens
    b = "b" * 32  # 8 tokens -> triggers else branch
    out = c.chunk_text(a + "\n\n" + b)
    # First chunk is purely 'a's
    assert out[0] == a
    # Second chunk should start with the 8-char overlap of 'a's then 'b's
    assert out[1].startswith("a" * (ov_tok * 4))
    assert set(out[1].replace("\n", "")) <= {"a", "b"}


def test_split_long_block_returns_original_when_short():
    # Cover `_split_long_block` early return branch `if n <= char_cap: return [block]`.
    max_tok = 10
    block = "x" * (max_tok * 4)  # exactly char_cap
    got = _split_long_block(block, max_tokens=max_tok, overlap_tokens=2)
    assert got == [block]


def test_split_long_block_overlap_ge_cap_slides_by_one():
    """
    Cover path where start advances by `start + 1` because overlap_chars >= char_cap.
    Expect 4 windows for len = char_cap + 3.
    """
    max_tok = 10
    char_cap = max_tok * 4
    block = "z" * (char_cap + 3)
    got = _split_long_block(block, max_tokens=max_tok, overlap_tokens=max_tok)  # overlap == cap
    assert len(got) == 4
    # Windows should be monotonically shifting by 1 char
    assert all(len(w) == char_cap or i == len(got) - 1 for i, w in enumerate(got))


def test_split_paragraphs_header_breaks_buffer():
    from corpusloom.chunking import _split_paragraphs

    text = "para line\n# Heading 1\nnext"
    blocks = _split_paragraphs(text)
    assert blocks == ["para line", "# Heading 1\nnext"]


def test_split_paragraphs_header_terminated_by_blank():
    text = "# Heading 1\n\nNext para line"
    blocks = _split_paragraphs(text)
    assert blocks == ["# Heading 1", "Next para line"]


def test_split_paragraphs_header_breaks_buffer():
    # Reload to avoid stale module copies between runs
    import importlib

    import corpusloom.chunking as ch

    importlib.reload(ch)

    text = "para line\n# Heading 1\nnext"
    blocks = ch._split_paragraphs(text)

    #  1) the paragraph before the header
    #  2) the header line + immediately following non-blank lines
    assert len(blocks) == 2, f"unexpected blocks: {blocks!r}"
    assert blocks[0].strip() == "para line", f"first block was {blocks[0]!r}"
    # Second block is the header *plus* the next line (until a blank line)
    assert blocks[1].splitlines() == ["# Heading 1", "next"], f"second block was {blocks[1]!r}"


def test_hard_wrap_space_cut_without_newline_exact_match():
    s = "word1 word2 word3"
    pieces = _hard_wrap(s, width_chars=10, overlap_chars=2)
    # First piece cuts at width (not at the early space), so it's exactly 10 chars.
    assert pieces[0] == "word1 word"
    # Second piece should finish the sentence cleanly.
    assert pieces[-1].endswith("word3")
    # No double-space artifacts in any piece.
    assert all("  " not in p for p in pieces)


def test_chunker_else_branch_hard_wrap_true_path_overlap_zero_no_tail():
    """
    Hit the else branch with hard-wrap True AND the 'no overlap' branch (ov_chars == 0),
    covering the case where `chunks and ov_chars > 0` is False.
    """
    c = Chunker(max_tokens=10, overlap_tokens=0)
    a = "x" * 35  # 9 tokens -> fills cur but doesn't flush yet
    b = "y" * 80  # 20 tokens -> triggers else and hard-wrap path
    out = c.chunk_text(a + "\n\n" + b)
    assert len(out) >= 2
    # Since overlap is zero, next chunk should not start with 'x'
    assert not out[1].startswith("x")


def test_chunker_flush_on_reaching_max_in_if_branch():
    # max_tokens=5; first block is exactly 5 tokens (20 chars) → append then immediate flush
    c = Chunker(max_tokens=5, overlap_tokens=2)
    block1 = "x" * 20  # ceil(20/4) == 5 tokens
    block2 = "y" * 4  # 1 token
    text = block1 + "\n\n" + block2

    out = c.chunk_text(text)
    # First chunk should be exactly the first block (flushed mid-loop)
    assert out[0] == block1
    # Second chunk contains only the second block content
    assert out[1].strip("y") == ""


def test_chunker_no_tail_overlap_when_overlap_zero():
    # Force the overflow/else branch, but with overlap disabled (ov_chars == 0)
    c = Chunker(max_tokens=5, overlap_tokens=0)
    a = "a" * 16  # ≈4 tokens
    b = "b" * 16  # ≈4 tokens
    text = a + "\n\n" + b

    out = c.chunk_text(text)
    # Must split into two chunks with NO overlap carried into the second one
    assert len(out) == 2
    assert out[0] == a
    assert out[1] == b  # no 'a' overlap at the head


def test_hard_wrap_short_string_breaks_immediately():
    # No '\n' or ' ' in range; rfind -> -1, so cut = j.
    # And since len(s) <= width, cut == n -> take the 'break' branch.
    s = "short"
    pieces = _hard_wrap(s, width_chars=50, overlap_chars=10)
    assert pieces == [s]


def test_chunker_else_branch_simple_path_with_overlap():
    """
    Hit 66->68 (else branch) with overlap enabled, and the simple append path
    (no hard-wrap since btoks <= 1.25*max).
    """
    c = Chunker(max_tokens=10, overlap_tokens=2)
    a = "x" * 35  # ≈ 9 tokens
    b = "y" * 12  # ≈ 3 tokens -> 9 + 3 > 10 triggers else; 3 <= 12.5 so simple append
    out = c.chunk_text(a + "\n\n" + b)

    # First chunk is the first block (flushed)
    assert out[0] == a
    # Second chunk starts with tail-overlap of 'a' and ends with 'b'
    assert out[1].startswith("x" * 8)  # ov_chars = 2*4 = 8 chars
    assert out[1].endswith(b)


def test_chunker_else_branch_without_overlap():
    """
    Hit 88->95 by making the overlap condition false (ov_chars == 0),
    so no tail-overlap is added after the flush.
    """
    c = Chunker(max_tokens=5, overlap_tokens=0)
    a = "a" * 16  # ≈ 4 tokens
    b = "b" * 16  # ≈ 4 tokens -> 4 + 4 > 5 triggers else; no overlap because ov==0
    out = c.chunk_text(a + "\n\n" + b)

    assert len(out) == 2
    assert out[0] == a
    assert out[1] == b  # no overlap prepended


def test_hard_wrap_continuation_branch_cut_lt_n():
    """
    Hit 118->128 in _hard_wrap by ensuring cut < n (i.e., do not break on the
    first iteration), so the loop continues and updates i.
    """
    # Place a newline near the right edge of width so cut > i + 0.6*width
    s = ("x" * 8) + "\n" + ("y" * 8) + "\n" + ("z" * 8)  # total 26 chars
    pieces = _hard_wrap(s, width_chars=10, overlap_chars=2)

    # First piece should end at the first newline (8 chars of 'x')
    assert pieces[0] == "x" * 8
    # We should have continued beyond the first piece (cut < n path)
    assert len(pieces) >= 2
    # And later pieces include y/z content
    assert any("y" in p for p in pieces)
    assert any("z" in p for p in pieces)


def test_first_block_uses_notcur_branch_even_when_too_big_codeblock():
    """
    Covers 66->68: in the main loop, the condition
      (cur_toks + btoks <= max) OR (not cur)
    is satisfied by the **not cur** part while btoks > max.
    We prevent early subdivision by using a fenced code block.
    """
    c = Chunker(max_tokens=10, overlap_tokens=2)
    code = "```txt\n" + ("x" * 200) + "\n```"  # huge code fence (exempt from subdivision)
    out = c.chunk_text(code)
    # Should produce at least one chunk containing the code fence
    assert any(seg.strip().startswith("```") and seg.strip().endswith("```") for seg in out)


def test_no_tail_overlap_when_overlap_disabled():
    """
    Covers 88->95: take the else branch (overflow with non-empty cur),
    but make ov_chars == 0 so the 'if chunks and ov_chars > 0' guard is false.
    """
    c = Chunker(max_tokens=5, overlap_tokens=0)  # ov_chars == 0
    a = "a" * 16  # ≈ 4 tokens
    b = "b" * 16  # ≈ 4 tokens -> overflow triggers else; no overlap appended
    out = c.chunk_text(a + "\n\n" + b)
    assert out[0] == a
    assert out[1] == b


def test_hard_wrap_breaks_when_segment_fits_exactly():
    """
    Covers 118->128 in _hard_wrap: choose a string whose first window sets cut == j == n,
    so 'if cut >= n: break' is taken.
    """
    s = "abcdefghij"  # length == width
    pieces = _hard_wrap(s, width_chars=10, overlap_chars=3)
    assert pieces == ["abcdefghij"]


def test_edge_66_68_or_short_circuit_first_block_big_code():
    """
    Hit the path where the OR condition is satisfied by 'not cur' while
    (cur_toks + btoks) > max_tokens, and then immediately flushes because
    cur_toks >= max_tokens. This exercises arc 66->68.
    """
    c = Chunker(max_tokens=10, overlap_tokens=2)
    # Code fences are never pre-split in the 'norm' stage, so btoks stays >> max.
    huge_code = "```txt\n" + ("Z" * 200) + "\n```"
    chunks = c.chunk_text(huge_code)
    # We should get at least one chunk that is the fenced block (added then flushed).
    assert any(seg.strip().startswith("```") and seg.strip().endswith("```") for seg in chunks)


def test_edge_88_95_overlap_branch_taken():
    """
    Force the overflow 'else' branch AND take the 'if chunks and ov_chars > 0' overlap path.
    This exercises arc 88->95.
    """
    # ov_chars = overlap_tokens*4 = 8, so we expect an 8-char overlap from the previous chunk.
    c = Chunker(max_tokens=10, overlap_tokens=2)
    a = "a" * 32  # ≈ 8 tokens -> first chunk content
    b = "b" * 32  # ≈ 8 tokens -> triggers overflow with non-empty cur
    out = c.chunk_text(a + "\n\n" + b)

    # After overflow, previous chunk 'a'*32 is flushed; overlap of 8 'a's is prefixed to the next cur.
    # So we should see an 'a'*32 chunk followed by a chunk that starts with at least 8 'a's
    # and then contains 'b's (the new block).
    assert a in out[0]
    assert any(seg.startswith("a" * 8) and "b" in seg for seg in out[1:])
