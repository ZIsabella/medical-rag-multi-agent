from sentence_transformers import SentenceTransformer

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # بارگذاری سنگین فقط یک‌بار
        _model = SentenceTransformer("BAAI/bge-m3")
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    return get_model().encode(
        texts,
        normalize_embeddings=True,   # مهم برای cosine
        batch_size=32,
        show_progress_bar=False,
    ).tolist()
