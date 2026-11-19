"""Model baseline FastText Subword Embedding."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import tokenize


class FastTextModel:
    """Implementasi FastText dengan mean-pooling representasi kata/subword."""

    def __init__(self, competencies_df: pd.DataFrame, model_path: str | Path = "models/fasttext.model"):
        self.competencies = competencies_df.copy().reset_index(drop=True)
        self.model_path = Path(model_path)
        self.ft_model = None
        self.competency_vectors = None
        if self.model_path.exists():
            self.load_model()

    def load_model(self):
        self.ft_model = FastText.load(str(self.model_path))
        self._compute_competency_vectors()

    def _get_text_vector(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        if not tokens or self.ft_model is None:
            return np.zeros(self.ft_model.vector_size if self.ft_model else 80, dtype=float)
        vectors = []
        for t in tokens:
            try:
                vectors.append(self.ft_model.wv[t])
            except KeyError:
                continue
        if not vectors:
            return np.zeros(self.ft_model.vector_size, dtype=float)
        mean_vec = np.mean(vectors, axis=0)
        norm = np.linalg.norm(mean_vec)
        return mean_vec / norm if norm > 0 else mean_vec

    def _compute_competency_vectors(self):
        vectors = []
        for _, row in self.competencies.iterrows():
            vectors.append(self._get_text_vector(row["teks_profil_lengkap"]))
        self.competency_vectors = np.array(vectors, dtype=float)

    def score_book(self, title: str, abstract: str) -> np.ndarray:
        """Menghitung cosine similarity FastText terhadap 67 kompetensi."""
        if self.competency_vectors is None:
            self._compute_competency_vectors()
        text = f"{title}. {abstract}"
        book_vec = self._get_text_vector(text).reshape(1, -1)
        sims = cosine_similarity(book_vec, self.competency_vectors)[0]
        return np.array(sims, dtype=float)

    def score_books_batch(self, books_df: pd.DataFrame) -> np.ndarray:
        """Menghitung matriks skor buku x kompetensi (N_books x 67)."""
        matrix = []
        for _, b in books_df.iterrows():
            matrix.append(self.score_book(b["judul"], b["abstrak"]))
        return np.array(matrix, dtype=float)

    def predict_competencies(
        self, title: str, abstract: str, top_k: int = 5, min_score: float = 0.0
    ) -> list[dict]:
        """Prediksi Top-K kompetensi untuk satu buku."""
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
                "model": "fasttext"
            })
        return results
