"""Model Cross-Encoder Reranker (BAAI/bge-reranker-base / bge-reranker-v2-m3)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.preprocessing import normalize_text


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class CrossEncoderRerankerModel:
    """Implementasi Cross-Encoder Reranker untuk pemeringkatan relevansi mendalam."""

    def __init__(
        self,
        competencies_df: pd.DataFrame,
        pretrained_id: str = "BAAI/bge-reranker-base",
        device: str = "cpu"
    ):
        self.competencies = competencies_df.copy().reset_index(drop=True)
        self.pretrained_id = pretrained_id
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.pretrained_id, device=self.device)
        return self._model

    def score_pairs(self, book_text: str, candidate_indices: list[int] | None = None) -> np.ndarray:
        """Menghitung skor probabilitas relevansi [0, 1] melalui cross-attention pasangan."""
        model = self._get_model()
        clean_book = normalize_text(book_text)

        if candidate_indices is None:
            indices = list(range(len(self.competencies)))
        else:
            indices = candidate_indices

        pairs = [
            (clean_book, normalize_text(self.competencies.iloc[i]["teks_profil_lengkap"]))
            for i in indices
        ]

        raw_scores = model.predict(pairs, show_progress_bar=False)
        raw_scores = np.array(raw_scores, dtype=float)

        # Jika output belum [0, 1], terapkan sigmoid
        if raw_scores.min() < 0.0 or raw_scores.max() > 1.0:
            probs = _sigmoid(raw_scores)
        else:
            probs = raw_scores

        full_scores = np.zeros(len(self.competencies), dtype=float)
        for idx, p in zip(indices, probs):
            full_scores[idx] = float(p)

        return full_scores

    def rerank_candidates(
        self,
        title: str,
        abstract: str,
        candidate_scores: np.ndarray,
        top_candidates: int = 15,
        top_k: int = 5
    ) -> list[dict]:
        """Dua tahap (Two-Stage): Ambil Top-N dari retriever, lalu rerank dengan Cross-Encoder."""
        top_indices = np.argsort(-candidate_scores)[:top_candidates].tolist()
        book_text = f"{title}. {abstract}"
        reranked_scores = self.score_pairs(book_text, candidate_indices=top_indices)

        order = np.argsort(-reranked_scores)
        results = []
        for rank, idx in enumerate(order[:top_k], start=1):
            s = float(reranked_scores[idx])
            comp = self.competencies.iloc[int(idx)]
            results.append({
                "rank": rank,
                "competency_id": comp["competency_id"],
                "competency_name": comp["competency_name"],
                "score": round(s, 4),
                "model": "bge_reranker"
            })
        return results

    def predict_competencies(
        self, title: str, abstract: str, top_k: int = 5, min_score: float = 0.0
    ) -> list[dict]:
        """Menilai langsung seluruh 67 kompetensi dengan Cross-Encoder."""
        book_text = f"{title}. {abstract}"
        scores = self.score_pairs(book_text)
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
                "model": "bge_reranker"
            })
        return results
