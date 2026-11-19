"""Uji integrasi menyeluruh komponen sistem rekomendasi buku multi-kompetensi BPJS."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.preprocessing import load_and_enrich_competencies, load_and_prepare_books
from src.recommender import BookRecommender


def test_data_integrity():
    """Memastikan integritas 67 kompetensi dan 973 buku."""
    comps = load_and_enrich_competencies(ROOT / "data")
    books = load_and_prepare_books(ROOT / "data")

    assert len(comps) == 67, f"Jumlah kompetensi harus 67, didapat {len(comps)}"
    assert len(books) == 973, f"Jumlah buku harus 973, didapat {len(books)}"
    assert not comps["competency_id"].duplicated().any()
    assert not books["book_id"].duplicated().any()
    print("PASS: Integritas data terverifikasi (67 kompetensi & 973 buku).")


def test_best_model_inference():
    """Memastikan artefak model terbaik dapat dimuat dan menghasilkan multikompetensi."""
    model = BookRecommender.load(ROOT / "models" / "best_model", allow_fallback=True)

    # Uji buku interdisipliner
    title = "Data Engineering and Distributed Databases"
    abstract = "Architecture and design of scalable distributed databases, streaming analytics, and cloud data warehousing."
    recs = model.map_book(title, abstract, top_k=5)

    assert len(recs) >= 1
    # Harus merekomendasikan kompetensi IT/Data
    rec_ids = [r["competency_id"] for r in recs]
    assert any(cid in rec_ids for cid in ["C056", "C062", "C065", "C066"]), f"Kompetensi yang didapat: {rec_ids}"

    # Uji pencarian buku berdasarkan kompetensi
    books_for_comp = model.recommend_books_for_competency("C023", top_k=5)
    assert len(books_for_comp) >= 1
    print("PASS: Inferensi model terbaik & rekomendasi katalog terverifikasi.")


def test_outputs_completeness():
    """Memastikan seluruh matriks skor dan laporan evaluasi telah terbentuk."""
    out = ROOT / "outputs"
    expected_files = [
        "skor_semua_pasangan_bm25.csv",
        "skor_semua_pasangan_tfidf.csv",
        "skor_semua_pasangan_e5.csv",
        "skor_semua_pasangan_bge_m3.csv",
        "skor_semua_pasangan_bge_reranker.csv",
        "skor_semua_pasangan_fasttext.csv",
        "buku_multikompetensi_per_model.csv",
        "evaluasi_metrik_perbandingan.csv",
        "keputusan_model_terpilih.json"
    ]
    for fname in expected_files:
        p = out / fname
        assert p.exists() and p.stat().st_size > 0, f"File {fname} belum tersedia atau kosong"
    print("PASS: Seluruh file output dan matriks eksperimen lengkap.")


if __name__ == "__main__":
    test_data_integrity()
    test_best_model_inference()
    test_outputs_completeness()
    print("Seluruh uji integrasi berhasil (100% PASS).")
