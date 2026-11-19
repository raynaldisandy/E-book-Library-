"""Model rekomendasi BM25 (Okapi BM25)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from src.preprocessing import tokenize, clean_text_for_lexical


class BM25Model:
    """Implementasi BM25 untuk pencocokan judul + abstrak buku ke 67 kompetensi."""

    def __init__(self, competencies_df: pd.DataFrame):
        self.competencies = competencies_df.copy().reset_index(drop=True)
        self.corpus_tokens = [
            tokenize(row["teks_profil_lengkap"])
            for _, row in self.competencies.iterrows()
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def score_book(self, title: str, abstract: str) -> np.ndarray:
        """Menghitung skor BM25 untuk 67 kompetensi terhadap teks buku."""
        query_text = f"{title}. {abstract}"
        query_tokens = tokenize(query_text)
        if not query_tokens:
            return np.zeros(len(self.competencies), dtype=float)
        scores = self.bm25.get_scores(query_tokens)
        return np.array(scores, dtype=float)

    def score_books_batch(self, books_df: pd.DataFrame) -> np.ndarray:
        """Menghitung matriks skor buku x kompetensi (shape: N_books x 67)."""
        matrix = []
        for _, b in books_df.iterrows():
            matrix.append(self.score_book(b["judul"], b["abstrak"]))
        return np.array(matrix, dtype=float)

    def predict_competencies(
        self, title: str, abstract: str, top_k: int = 5, min_score: float = 0.0
    ) -> list[dict]:
        """Prediksi Top-K kompetensi untuk satu buku baru."""
        scores = self.score_book(title, abstract)
        order = np.argsort(-scores)
        results = []
        for rank, idx in enumerate(order[:top_k], start=1):
            s = float(scores[idx])
            if s < min_score:
                continue
            comp = self.competencies.iloc[int(idx)]
            results.append({
                "rank": rank,
                "competency_id": comp["competency_id"],
                "competency_name": comp["competency_name"],
                "score": round(s, 4),
                "model": "bm25"
            })
        return results
