"""Offline checks for chunking + citation verification. Run: python test_rag.py (or pytest)."""
from rag import BilingualTokenizer, _norm, _vid, chunk_text, n_tokens, verify_citations


def test_chunking():
    text = "\n\n".join(f"Paragraph {i}. " + "Some filler sentence here. " * 20 for i in range(30))
    chunks = chunk_text(text, max_tokens=200, overlap=40)
    assert len(chunks) > 1
    assert all(n_tokens(c) <= 200 for c in chunks)
    # every paragraph survives somewhere
    assert all(any(f"Paragraph {i}." in c for c in chunks) for i in range(30))
    # one giant sentence gets hard-split, still under limit
    assert all(n_tokens(c) <= 50 for c in chunk_text("word " * 500, max_tokens=50, overlap=10))
    assert chunk_text("") == []


def test_thai_chunking():
    # no periods, space-separated Thai sentences, one long paragraph
    th = " ".join(["บริษัทมีนโยบายการคืนสินค้าภายในสามสิบวันนับจากวันที่ได้รับสินค้า"] * 60)
    chunks = chunk_text(th, max_tokens=120, overlap=20)
    assert len(chunks) > 1 and all(n_tokens(c) <= 120 for c in chunks)
    assert all("\ufffd" not in c for c in chunks)  # never split a Thai char's bytes
    # one unbroken Thai run (no spaces) still splits on word boundaries
    run = chunk_text("การคืนสินค้า" * 200, max_tokens=80, overlap=0)
    assert len(run) > 1 and all("\ufffd" not in c and n_tokens(c) <= 80 for c in run)


def test_bilingual_tokenizer():
    tok = BilingualTokenizer()("นโยบายการคืนสินค้าภายใน 30 วัน Refund policies")
    assert "สินค้า" in tok and "นโยบาย" in tok  # Thai segmented into words
    assert "refund" in tok and "polici" in tok  # English lowercased + stemmed
    assert "ภายใน" not in tok  # Thai stopword removed


def test_ids_ascii():
    assert _vid("คู่มือ#1.md").isascii() and "#" not in _vid("a#b")
    assert _vid("a") != _vid("ab") and len(_vid("ก" * 200)) <= 480


def test_citations():
    texts = ["Pinecone supports sparse-dense vectors with the dotproduct metric.", "GPT-4o was released in 2024."]
    ok = verify_citations("Use dotproduct [1]. Released 2024 [2].", [
        {"source_id": 1, "quote": "supports sparse-dense vectors with the dotproduct metric"},
        {"source_id": 2, "quote": "GPT-4o was released in 2024"},
    ], texts)
    assert ok["verified"] and ok["unsupported_markers"] == []

    fake = verify_citations("Released 2023 [2].", [{"source_id": 2, "quote": "released in 2023"}], texts)
    assert not fake["verified"] and fake["unsupported_markers"] == [2]

    missing = verify_citations("Claim [1][3].", [{"source_id": 1, "quote": "dotproduct metric"}], texts)
    assert not missing["verified"] and missing["unsupported_markers"] == [3]

    out_of_range = verify_citations("X [9].", [{"source_id": 9, "quote": "anything"}], texts)
    assert not out_of_range["verified"]

    empty_quote = verify_citations("X [1].", [{"source_id": 1, "quote": "  "}], texts)
    assert not empty_quote["verified"]
    trivial = verify_citations("X [1].", [{"source_id": 1, "quote": "the"}], texts)
    assert not trivial["verified"]


def test_thai_citations():
    assert _norm("ที่ทีม") == "ที่ทีม"  # vowel/tone marks kept
    texts = ["บริษัทมีนโยบายการคืนสินค้าภายใน 30 วัน นับจากวันที่ได้รับสินค้า"]
    ok = verify_citations("คืนสินค้าได้ภายใน 30 วัน [1]", [{"source_id": 1, "quote": "การคืนสินค้า ภายใน 30 วัน"}], texts)
    assert ok["verified"]  # spacing differences tolerated
    bad = verify_citations("คืนสินค้าได้ภายใน 60 วัน [1]", [{"source_id": 1, "quote": "การคืนสินค้าภายใน 60 วัน"}], texts)
    assert not bad["verified"]
    # Thai question answered from English source: quote stays in source language
    mixed = verify_citations("ใช้ dotproduct [1]", [{"source_id": 1, "quote": "dotproduct metric"}],
                             ["Use the dotproduct metric."])
    assert mixed["verified"]


if __name__ == "__main__":
    test_chunking()
    test_thai_chunking()
    test_bilingual_tokenizer()
    test_ids_ascii()
    test_citations()
    test_thai_citations()
    print("ok")
