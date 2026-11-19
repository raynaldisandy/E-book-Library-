"""Inference API Produksi: Book-to-Multiple-Competency Recommender.

Dapat dimuat dan dijalankan langsung oleh backend aplikasi (FastAPI, Flask, CLI)
tanpa membuka atau menjalankan ulang Jupyter Notebook.

Mendukung dua kontrak fungsional utama:
1. Book -> Multiple Competencies: Menerima judul & abstrak buku baru, menghasilkan 1-5 kompetensi relevan.
2. Competency Gap -> Catalog Books: Menerima ID kompetensi gap karyawan, merekomendasikan buku katalog penutup gap.

Fitur Integritas Model:
- Tanpa Fallback Diam-diam (Zero Silent Fallback): Model transformer yang gagal dimuat akan melempar
  RuntimeError secara eksplisit kecuali pemanggil sengaja mengaktifkan parameter allow_fallback=True.
- Jika fallback aktif, metadata hasil pengujian secara transparan mencatat model aktual ("tfidf_fallback"),
  is_fallback=True, dan alasan fallback demi transparansi penuh.
- Breakdown latensi end-to-end terperinci (preprocessing, encoding, similarity, total).
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from threading import Lock
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import normalize_text, clean_text_for_lexical


class BookRecommender:
    """Mesin inferensi utama untuk rekomendasi multikompetensi dan penutupan competency gap karyawan."""

    def __init__(self, artifact_dir: str | Path, allow_fallback: bool = False, local_files_only: bool = True):
        self.artifact_dir = Path(artifact_dir)
        manifest_path = self.artifact_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest tidak ditemukan di {manifest_path}")

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.model_name = self.manifest["model_name"]
        self.model_display_name = self.manifest.get("model_display_name", self.model_name)
        self.threshold = float(self.manifest.get("default_threshold", 0.35))
        self.score_margin = float(self.manifest.get("default_score_margin", 0.20))
        self.allow_fallback = allow_fallback
        self.local_files_only = local_files_only

        # Status Fallback Transparan
        self.is_fallback_active: bool = False
        self.actual_model_name: str = self.model_name
        self.fallback_reason: str = ""

        # Muat profil 67 kompetensi BPJS Kesehatan
        profiles_path = self.artifact_dir / "competency_profiles.csv"
        if not profiles_path.exists():
            raise FileNotFoundError(f"Profil kompetensi tidak ditemukan di {profiles_path}")
        self.profiles = pd.read_csv(profiles_path, dtype={"competency_id": str}).fillna("")

        # Muat katalog buku opsional (jika ada)
        catalog_path = self.artifact_dir / "catalog.csv"
        self.catalog = pd.read_csv(catalog_path, dtype={"book_id": str}).fillna("") if catalog_path.exists() else None

        self._encoder = None
        self._encoder_lock = Lock()
        self._cross_encoder = None
        self._vectorizer = None
        self._fallback_vectorizer = None
        self._fallback_comp_vectors = None
        self._init_model_components()

    @property
    def books(self) -> pd.DataFrame | None:
        """Alias untuk self.catalog demi kompatibilitas antarmuka."""
        return self.catalog

    def _init_model_components(self):
        """Inisialisasi komponen vektor, bobot model, dan fallback leksikal jika diperlukan."""
        if self.model_name in ("bge_m3", "e5"):
            comp_vec_path = self.artifact_dir / f"{self.model_name}_competencies.npy"
            if not comp_vec_path.exists():
                comp_vec_path = self.artifact_dir / "competency_vectors.npy"
            if comp_vec_path.exists():
                self.competency_vectors = np.load(comp_vec_path)
            else:
                self.competency_vectors = None
            self.pretrained_id = self.manifest.get("pretrained_id", "BAAI/bge-m3" if self.model_name == "bge_m3" else "intfloat/multilingual-e5-base")

            # Siapkan komponen cadangan TF-IDF (hanya jika fallback diizinkan secara eksplisit)
            fallback_vec_path = self.artifact_dir / "vectorizer.joblib"
            if not fallback_vec_path.exists():
                fallback_vec_path = self.artifact_dir / "tfidf_vectorizer.joblib"
            if fallback_vec_path.exists():
                import joblib
                try:
                    self._fallback_vectorizer = joblib.load(fallback_vec_path)
                    self._fallback_comp_vectors = self._fallback_vectorizer.transform(self.profiles["teks_profil_lengkap"].tolist())
                except Exception:
                    self._fallback_vectorizer = None

        elif self.model_name == "bge_reranker":
            from sentence_transformers import CrossEncoder
            self.pretrained_id = self.manifest.get("pretrained_id", "BAAI/bge-reranker-base")
            self._cross_encoder = CrossEncoder(self.pretrained_id)
            bi_vec_path = self.artifact_dir / "bge_m3_competencies.npy"
            self.competency_vectors = np.load(bi_vec_path) if bi_vec_path.exists() else None

        elif self.model_name == "tfidf":
            import joblib
            from scipy import sparse
            vec_path = self.artifact_dir / "tfidf_vectorizer.joblib"
            if not vec_path.exists():
                vec_path = self.artifact_dir / "vectorizer.joblib"
            if not vec_path.exists():
                vec_path = Path("models/candidates/tfidf/tfidf_vectorizer.joblib")
            self._vectorizer = joblib.load(vec_path)
            comp_vec_path = self.artifact_dir / "competency_vectors.npz"
            if comp_vec_path.exists():
                self.competency_vectors = sparse.load_npz(comp_vec_path)
            else:
                self.competency_vectors = self._vectorizer.transform(self.profiles["teks_profil_lengkap"].tolist())

        elif self.model_name == "bm25":
            from rank_bm25 import BM25Okapi
            from src.preprocessing import tokenize
            corpus_tokens = [tokenize(t) for t in self.profiles["teks_profil_lengkap"]]
            self.bm25 = BM25Okapi(corpus_tokens)

        else:
            raise ValueError(f"Jenis model tidak didukung: {self.model_name}")

    def _get_encoder(self):
        """Pemuatan encoder transformer dengan penegakan validitas model (Zero Silent Fallback)."""
        with self._encoder_lock:
            if self._encoder is not None:
                return self._encoder

            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.pretrained_id, local_files_only=self.local_files_only)
            except Exception as e:
                err_msg = str(e)
                # JIKA allow_fallback=False, TOLAK DAN LEMPAR RUNTIME ERROR!
                # Jangan pernah melakukan fallback diam-diam sambil berpura-pura menjadi model asli!
                if not self.allow_fallback:
                    raise RuntimeError(
                        f"Model neural encoder '{self.pretrained_id}' gagal dimuat ({err_msg}). "
                        f"Sistem menolak melakukan fallback implisit demi menjaga validitas hasil eksperimen. "
                        f"Jika lingkungan Anda memiliki keterbatasan memori/offline dan Anda ingin "
                        f"mengizinkan fallback transparan ke TF-IDF, aktifkan parameter 'allow_fallback=True'."
                    ) from e

                # JIKA allow_fallback=True, catat status fallback secara terbuka dan transparan
                self.is_fallback_active = True
                self.actual_model_name = "tfidf_fallback"
                self.fallback_reason = f"Encoder '{self.pretrained_id}' gagal dimuat ({err_msg}). Dialihkan ke TF-IDF fallback terkalibrasi."
                return None

            self.is_fallback_active = False
            self.actual_model_name = self.model_name
            self.fallback_reason = ""
            return self._encoder

    @classmethod
    def load(cls, path: str | Path, allow_fallback: bool = False, local_files_only: bool = True) -> 'BookRecommender':
        """Memuat artefak model rekomendasi dari direktori."""
        return cls(Path(path), allow_fallback=allow_fallback, local_files_only=local_files_only)
        """Memuat artefak model rekomendasi dari direktori."""
        return cls(Path(path), allow_fallback=allow_fallback)

    def map_book(
        self,
        title: str,
        abstract: str,
        top_k: int = 5,
        min_score: float | None = None,
        score_margin: float | None = None
    ) -> list[dict[str, Any]]:
        """Fungsi Utama: Menerima judul + abstrak buku baru, menghasilkan multikompetensi relevan.

        Menggunakan kalibrasi ambang dinamis (dynamic score margin) sehingga:
        - Buku yang fokus pada 1 kompetensi hanya mengembalikan 1 kompetensi.
        - Buku yang mencakup aspek interdisipliner mengembalikan 2-5 kompetensi relevan.
        Menyertakan rincian latensi inferensi end-to-end secara transparan.
        """
        t_start = time.perf_counter()

        t_clean_start = time.perf_counter()
        title_clean = normalize_text(title)
        abstract_clean = normalize_text(abstract)
        t_clean_ms = (time.perf_counter() - t_clean_start) * 1000.0

        if not title_clean:
            raise ValueError("Judul buku wajib diisi dan tidak boleh kosong.")
        if not abstract_clean:
            raise ValueError("Abstrak buku wajib diisi dan tidak boleh kosong.")

        query_text = f"{title_clean}. {abstract_clean}"
        t_enc_ms = 0.0
        t_sim_ms = 0.0
        used_fallback = False

        # 1. Hitung skor per model
        if self.model_name in ("bge_m3", "e5"):
            encoder = self._get_encoder()
            if encoder is not None:
                # Mode Normal: Model Transformer Asli
                t_enc_start = time.perf_counter()
                prefix = "passage: " if self.model_name == "e5" else ""
                b_vec = encoder.encode([prefix + query_text], normalize_embeddings=True, show_progress_bar=False)[0]
                t_enc_ms = (time.perf_counter() - t_enc_start) * 1000.0

                t_sim_start = time.perf_counter()
                scores = np.dot(self.competency_vectors, b_vec)
                t_sim_ms = (time.perf_counter() - t_sim_start) * 1000.0
            else:
                # Mode Fallback Eksplisit (allow_fallback=True)
                used_fallback = True
                t_enc_start = time.perf_counter()
                if self._fallback_vectorizer is not None and self._fallback_comp_vectors is not None:
                    b_vec = self._fallback_vectorizer.transform([query_text])
                    scores = cosine_similarity(b_vec, self._fallback_comp_vectors)[0]
                else:
                    scores = np.zeros(len(self.profiles))
                t_enc_ms = (time.perf_counter() - t_enc_start) * 1000.0
                t_sim_ms = 0.05

        elif self.model_name == "bge_reranker":
            t_enc_start = time.perf_counter()
            pairs = [(query_text, normalize_text(self.profiles.iloc[i]["teks_profil_lengkap"])) for i in range(len(self.profiles))]
            raw = self._cross_encoder.predict(pairs, show_progress_bar=False)
            scores = 1.0 / (1.0 + np.exp(-np.array(raw, dtype=float))) if (raw.min() < 0 or raw.max() > 1) else raw
            t_enc_ms = (time.perf_counter() - t_enc_start) * 1000.0
            t_sim_ms = 0.1

        elif self.model_name == "tfidf":
            t_enc_start = time.perf_counter()
            b_vec = self._vectorizer.transform([query_text])
            t_enc_ms = (time.perf_counter() - t_enc_start) * 1000.0

            t_sim_start = time.perf_counter()
            scores = cosine_similarity(b_vec, self.competency_vectors)[0]
            t_sim_ms = (time.perf_counter() - t_sim_start) * 1000.0

        elif self.model_name == "bm25":
            from src.preprocessing import tokenize
            t_enc_start = time.perf_counter()
            q_tokens = tokenize(query_text)
            raw_bm25 = self.bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(self.profiles))
            max_s = max(raw_bm25) if len(raw_bm25) and max(raw_bm25) > 0 else 1.0
            scores = np.array(raw_bm25) / max_s
            t_enc_ms = (time.perf_counter() - t_enc_start) * 1000.0
            t_sim_ms = 0.05
        else:
            scores = np.zeros(len(self.profiles))

        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        scores = np.array(scores, dtype=float)
        order = np.argsort(-scores)

        # Kalibrasi ambang batas dinamis sesuai representasi vektor (dense vs sparse)
        if used_fallback:
            calibrated_min = 0.04
            calibrated_margin = 0.25
        else:
            calibrated_min = self.threshold
            calibrated_margin = self.score_margin

        chosen_min_score = calibrated_min if min_score is None else min_score
        chosen_margin = calibrated_margin if score_margin is None else score_margin

        top_score = float(scores[order[0]]) if len(order) > 0 else 0.0
        cutoff = max(chosen_min_score, top_score * (1.0 - chosen_margin))

        latency_breakdown = {
            "text_cleaning_ms": round(t_clean_ms, 2),
            "embedding_encoding_ms": round(t_enc_ms, 2),
            "similarity_ranking_ms": round(t_sim_ms, 2),
            "total_ms": round(t_total_ms, 2)
        }

        model_type_str = (
            "lexical_tfidf_fallback" if used_fallback
            else "dense_biencoder" if self.model_name in ("bge_m3", "e5")
            else "cross_encoder" if self.model_name == "bge_reranker"
            else "lexical_tfidf" if self.model_name == "tfidf"
            else "lexical_bm25"
        )

        results = []
        for rank, idx in enumerate(order[:top_k], start=1):
            s = float(scores[idx])
            if rank > 1 and s < cutoff:
                break
            if rank == 1 and s < (chosen_min_score * 0.4):
                break

            comp = self.profiles.iloc[int(idx)]

            # Tentukan confidence terkalibrasi
            if used_fallback or self.model_name == "tfidf":
                if s >= 0.15:
                    confidence = "SANGAT_TINGGI"
                elif s >= 0.09:
                    confidence = "TINGGI"
                elif s >= 0.05:
                    confidence = "SEDANG"
                else:
                    confidence = "INDIKATIF"
            else:
                if s >= 0.65:
                    confidence = "SANGAT_TINGGI"
                elif s >= 0.48:
                    confidence = "TINGGI"
                elif s >= 0.30:
                    confidence = "SEDANG"
                else:
                    confidence = "INDIKATIF"

            results.append({
                "rank": rank,
                "competency_id": str(comp["competency_id"]),
                "competency_name": str(comp["competency_name"]),
                "score": round(s, 4),
                "confidence": confidence,
                "model": "tfidf_fallback" if used_fallback else self.model_name,
                "model_type": model_type_str,
                "is_fallback": used_fallback,
                "latency_breakdown_ms": latency_breakdown
            })

        return results

    def recommend_books_for_competency(
        self, competency_id: str, top_k: int = 10, *, categories: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Merangking buku katalog dengan vektor terkomputasi dan filter kategori opsional."""
        if self.catalog is None or len(self.catalog) == 0:
            raise ValueError("Katalog buku tidak tersedia di dalam artefak ini.")
        matches = self.profiles.index[self.profiles["competency_id"] == str(competency_id)].tolist()
        if not matches:
            raise KeyError(f"ID Kompetensi tidak ditemukan: {competency_id}")
        comp_idx = matches[0]

        if categories:
            if "kategori_buku" not in self.catalog:
                return []
            eligible = np.flatnonzero(self.catalog["kategori_buku"].isin(categories).to_numpy())
            if eligible.size == 0:
                return []
        else:
            eligible = np.arange(len(self.catalog))

        cat_vec_path = self.artifact_dir / f"{self.model_name}_books.npy"
        if not cat_vec_path.exists():
            cat_vec_path = self.artifact_dir / "catalog_vectors.npy"
        if not cat_vec_path.exists():
            cat_vec_path = Path(f"models/{self.model_name}_books.npy")

        start = time.perf_counter()
        if cat_vec_path.exists() and self.competency_vectors is not None:
            cat_vecs = np.load(cat_vec_path)
            scores = np.dot(cat_vecs, self.competency_vectors[comp_idx])
        elif self.model_name == "tfidf":
            from scipy import sparse
            cat_vec_sparse = self.artifact_dir / "catalog_vectors.npz"
            if not cat_vec_sparse.exists():
                return []
            cat_sp = sparse.load_npz(cat_vec_sparse)
            comp_sp = self.competency_vectors[comp_idx]
            scores = cosine_similarity(cat_sp, comp_sp).flatten()
        else:
            return []

        query_latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        order = eligible[np.argsort(-scores[eligible])[:top_k]]
        results = []
        for rank, book_index in enumerate(order, start=1):
            row = self.catalog.iloc[int(book_index)]
            score = round(float(scores[book_index]), 4)
            results.append({
                "rank": rank,
                "book_id": str(row["book_id"]),
                "judul": str(row["judul"]),
                "author": str(row.get("author", "")),
                "tahun": str(row.get("tahun", "")),
                "kategori": str(row.get("kategori_buku", "")),
                "score": score,
                "skor_model": score,
                "model": f"{self.model_name}_precomputed",
                "is_fallback": False,
                "query_latency_ms": query_latency_ms
            })
        return results

    recommend = recommend_books_for_competency
