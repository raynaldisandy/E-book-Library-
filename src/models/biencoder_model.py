"""Model rekomendasi Dense Bi-Encoder Transformer (Multilingual E5 dan BGE-M3)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import normalize_text


class DenseBiEncoderModel:
    """Implementasi Dense Bi-Encoder (Multilingual E5 / BGE-M3)."""

    def __init__(
        self,
        model_name: str,
        competencies_df: pd.DataFrame,
        pretrained_id: str | None = None,
        competency_vectors: np.ndarray | None = None,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.competencies = competencies_df.copy().reset_index(drop=True)
        self.pretrained_id = pretrained_id or (
            "intfloat/multilingual-e5-base" if model_name == "e5" else "BAAI/bge-m3"
        )
        self.competency_vectors = competency_vectors
        self.device = device
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.pretrained_id, device=self.device)
        return self._encoder

    def encode_competencies(self) -> np.ndarray:
        """Mengkodekan 67 teks profil kompetensi menjadi representasi dense ter-normalisasi."""
        encoder = self._get_encoder()
        texts = []
        for _, row in self.competencies.iterrows():
            t = normalize_text(row["teks_profil_lengkap"])
            prefix = "query: " if self.model_name == "e5" else ""
            texts.append(prefix + t)
        vecs = encoder.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        self.competency_vectors = np.array(vecs, dtype=np.float32)
        return self.competency_vectors

    def encode_text(self, text: str) -> np.ndarray:
        """Mengkodekan teks buku baru (passage: untuk E5)."""
        encoder = self._get_encoder()
        clean = normalize_text(text)
        prefix = "passage: " if self.model_name == "e5" else ""
        vec = encoder.encode([prefix + clean], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.array(vec, dtype=np.float32)

    def score_book(self, title: str, abstract: str, book_vector: np.ndarray | None = None) -> np.ndarray:
        """Menghitung cosine similarity terhadap 67 kompetensi."""
        if self.competency_vectors is None:
            self.encode_competencies()
        if book_vector is None:
            book_vector = self.encode_text(f"{title}. {abstract}")
        # Cosine similarity pada vektor ternormalisasi adalah dot product
        scores = np.dot(self.competency_vectors, book_vector)
        return np.array(scores, dtype=float)

    def score_books_batch(self, book_vectors: np.ndarray) -> np.ndarray:
        """Menghitung matriks skor buku x kompetensi (N_books x 67)."""
        if self.competency_vectors is None:
            self.encode_competencies()
        # book_vectors: (N_books, D), competency_vectors: (67, D)
        # matrix = book_vectors @ competency_vectors.T (N_books, 67)
        matrix = np.dot(book_vectors, self.competency_vectors.T)
        return np.array(matrix, dtype=float)

    def predict_competencies(
        self,
        title: str,
        abstract: str,
        top_k: int = 5,
        min_score: float = 0.0,
        book_vector: np.ndarray | None = None
    ) -> list[dict]:
        """Prediksi Top-K kompetensi untuk satu buku."""
        scores = self.score_book(title, abstract, book_vector=book_vector)
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
                "model": self.model_name
            })
        return results

    def save(self, output_dir: str | Path):
        """Menyimpan vektor kompetensi dan konfigurasi."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.competency_vectors is not None:
            np.save(out / f"{self.model_name}_competencies.npy", self.competency_vectors)

    @classmethod
    def load_precomputed(
        cls,
        model_name: str,
        competencies_df: pd.DataFrame,
        vectors_path: str | Path,
        pretrained_id: str | None = None
    ) -> 'DenseBiEncoderModel':
        vecs = np.load(vectors_path)
        return cls(
            model_name=model_name,
            competencies_df=competencies_df,
            pretrained_id=pretrained_id,
            competency_vectors=vecs
        )
