"""Model rekomendasi TF-IDF (Term Frequency - Inverse Document Frequency)."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import clean_text_for_lexical


class TFIDFModel:
    """Implementasi TF-IDF dengan n-gram (1,2) dan sublinear TF."""

    def __init__(self, competencies_df: pd.DataFrame, vectorizer: TfidfVectorizer | None = None):
        self.competencies = competencies_df.copy().reset_index(drop=True)
        self.vectorizer = vectorizer
        self.competency_vectors = None
        if self.vectorizer is not None:
            self._fit_competency_vectors()

    def fit(self, all_texts: list[str]) -> 'TFIDFModel':
        """Melatih TfidfVectorizer pada seluruh korpus buku dan profil kompetensi."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.85,
            preprocessor=clean_text_for_lexical
        )
        self.vectorizer.fit(all_texts)
        self._fit_competency_vectors()
        return self

    def _fit_competency_vectors(self):
        texts = self.competencies["teks_profil_lengkap"].tolist()
        self.competency_vectors = self.vectorizer.transform(texts)

    def score_book(self, title: str, abstract: str) -> np.ndarray:
        """Menghitung skor cosine similarity TF-IDF terhadap 67 kompetensi."""
        if self.vectorizer is None or self.competency_vectors is None:
            raise ValueError("Model belum dilatih!")
        text = f"{title}. {abstract}"
        book_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(book_vec, self.competency_vectors)[0]
        return np.array(sims, dtype=float)

    def score_books_batch(self, books_df: pd.DataFrame) -> np.ndarray:
        """Menghitung matriks skor buku x kompetensi (N_books x 67)."""
        texts = (books_df["judul"] + ". " + books_df["abstrak"]).tolist()
        books_vec = self.vectorizer.transform(texts)
        sim_matrix = cosine_similarity(books_vec, self.competency_vectors)
        return np.array(sim_matrix, dtype=float)

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
                "model": "tfidf"
            })
        return results

    def save(self, output_dir: str | Path):
        """Menyimpan artefak TF-IDF."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, out / "tfidf_vectorizer.joblib")
        if self.competency_vectors is not None:
            from scipy import sparse
            sparse.save_npz(out / "competency_vectors.npz", self.competency_vectors)

    @classmethod
    def load(cls, output_dir: str | Path, competencies_df: pd.DataFrame) -> 'TFIDFModel':
        """Memuat artefak TF-IDF yang tersimpan."""
        out = Path(output_dir)
        vec = joblib.load(out / "tfidf_vectorizer.joblib")
        instance = cls(competencies_df, vectorizer=vec)
        comp_vec_path = out / "competency_vectors.npz"
        if comp_vec_path.exists():
            from scipy import sparse
            instance.competency_vectors = sparse.load_npz(comp_vec_path)
        else:
            instance._fit_competency_vectors()
        return instance
