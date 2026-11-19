"""Pintu masuk sederhana untuk tim aplikasi."""
from pathlib import Path
from .recommender import BookRecommender


def load_candidate(project_root: str | Path, name: str = 'tfidf') -> BookRecommender:
    """Memuat artefak kandidat, bukan menyatakan model terbaik."""
    return BookRecommender.load(Path(project_root) / 'models' / 'candidates' / name)
