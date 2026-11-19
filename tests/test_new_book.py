"""Uji integrasi sistem rekomendasi buku untuk mengatasi competency gap karyawan BPJS Kesehatan.

Sistem ini berfungsi untuk:
1. Menentukan kompetensi karyawan yang dapat dikembangkan oleh suatu buku baru (Book -> Multiple Competencies).
2. Merekomendasikan buku katalog bagi karyawan yang memiliki gap kompetensi tertentu (Competency Gap -> Books).
3. Integritas Provenance Model: Menjamin ketiadaan fallback diam-diam (Zero Silent Fallback) dan transparansi metadata.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommender import BookRecommender


def test_new_book_best_model():
    """Uji inferensi model terbaik (models/best_model) untuk pemetaan kompetensi pengembangan pegawai."""
    best_model_path = ROOT / "models" / "best_model"

    # 1. Uji Kontrak Zero Silent Fallback (Strict Mode)
    model_strict = BookRecommender.load(best_model_path, allow_fallback=False)
    try:
        res_strict = model_strict.map_book("Test Title", "Test Abstract")
        assert res_strict[0]["model"] == "bge_m3", "Pada mode tanpa fallback, model harus bge_m3 murni"
        assert res_strict[0]["is_fallback"] is False
    except RuntimeError as e:
        # Jika lingkungan offline / bobot transformer lokal belum ada, RuntimeError WAJIB dilempar!
        assert "gagal dimuat" in str(e).lower() or "transformer" in str(e).lower()
        print("VERIFIED: Strict mode (allow_fallback=False) melempar RuntimeError secara eksplisit (Zero Silent Fallback).")

    # 2. Uji Mode Fallback Transparan (allow_fallback=True)
    model = BookRecommender.load(best_model_path, allow_fallback=True)

    # Input Bahasa Inggris (Healthcare Risk Management - menutup gap C023)
    title_en = "Healthcare Risk Mitigation Handbook"
    abstract_en = "This handbook discusses risk identification, operational risk assessment, monitoring and mitigation in health services."
    result_en = model.map_book(title_en, abstract_en, top_k=5)

    assert len(result_en) >= 1, "Rekomendasi tidak boleh kosong"
    assert result_en[0]["competency_id"] == "C023", f"Top kompetensi seharusnya Risk Management (C023), didapat {result_en[0]['competency_id']}"
    assert all("competency_id" in row and "competency_name" in row and "score" in row for row in result_en)
    assert all("latency_breakdown_ms" in row for row in result_en)

    # Validasi transparansi model yang sebenarnya berjalan
    if result_en[0]["is_fallback"]:
        assert result_en[0]["model"] == "tfidf_fallback"
        assert result_en[0]["model_type"] == "lexical_tfidf_fallback"
    else:
        assert result_en[0]["model"] == "bge_m3"
        assert result_en[0]["model_type"] == "dense_biencoder"

    # 3. Input Bahasa Indonesia (Audit & Manajemen Risiko Pegawai)
    title_id = "Tata Kelola Manajemen Risiko Operasional dan Pengawasan Internal Rumah Sakit"
    abstract_id = "Panduan praktis mitigasi risiko organisasi, audit operasional fasilitas kesehatan, dan kepatuhan tata kelola internal."
    result_id = model.map_book(title_id, abstract_id, top_k=5)

    assert len(result_id) >= 1, "Rekomendasi bahasa Indonesia tidak boleh kosong"
    rec_ids = [r["competency_id"] for r in result_id]
    assert any(cid in rec_ids for cid in ("C023", "C022", "C021", "C024", "C025")), f"Kompetensi yang didapat: {rec_ids}"

    # 4. Rekomendasi Buku untuk Karyawan yang Memiliki Competency Gap (C023 - Risk Management)
    gap_comp_id = "C023"
    books_for_gap = model.recommend_books_for_competency(gap_comp_id, top_k=5)
    assert len(books_for_gap) == 5, "Harus mengembalikan 5 rekomendasi buku untuk menutup gap kompetensi"
    assert all("judul" in b and "score" in b for b in books_for_gap)
    assert all(b["model"] == "bge_m3_precomputed" for b in books_for_gap), "Pencarian katalog harus menggunakan embedding BGE-M3 terkomputasi"
    assert all(b["is_fallback"] is False for b in books_for_gap)

    # 5. Validasi input kosong (kontrak input wajib judul dan abstrak)
    try:
        model.map_book("", "Abstrak saja")
        assert False, "Harus melempar ValueError jika judul kosong"
    except ValueError:
        pass

    print("PASS: test_new_book_best_model (Competency Gap & Multi-Competency) berhasil diverifikasi.")


def test_new_book_tfidf_candidate():
    """Uji inferensi model kandidat TF-IDF (models/candidates/tfidf) sebagai offline fallback mandiri."""
    candidate_path = ROOT / "models" / "candidates" / "tfidf"
    model = BookRecommender.load(candidate_path)

    title = "Healthcare Risk Mitigation Handbook"
    abstract = "This handbook discusses risk identification, assessment, monitoring and mitigation in health services."
    result = model.map_book(title, abstract, top_k=5)

    assert len(result) >= 1
    assert result[0]["competency_id"] == "C023"
    assert result[0]["model"] == "tfidf"
    assert result[0]["is_fallback"] is False
    assert result[0]["model_type"] == "lexical_tfidf"

    print("PASS: test_new_book_tfidf_candidate berhasil diverifikasi.")


def test_recovered_encoder_reports_dense_model(monkeypatch):
    model = BookRecommender.load(ROOT / "models" / "best_model", allow_fallback=True)
    title = "Healthcare Risk Mitigation Handbook"
    abstract = "This handbook discusses risk identification and mitigation in health services."

    def unavailable(*args, **kwargs):
        raise OSError("Weights have not finished downloading")

    dependency = SimpleNamespace(SentenceTransformer=unavailable)
    monkeypatch.setitem(sys.modules, "sentence_transformers", dependency)
    results = model.map_book(title, abstract)
    assert results and results[0]["is_fallback"] is True

    def available(*args, **kwargs):
        return SimpleNamespace(encode=lambda *args, **kwargs: [model.competency_vectors[0]])

    dependency.SentenceTransformer = available
    results = model.map_book(title, abstract)
    assert results and results[0]["model"] == "bge_m3"
    assert results[0]["model_type"] == "dense_biencoder"
    assert results[0]["is_fallback"] is False
    assert model.actual_model_name == "bge_m3"
    assert model.fallback_reason == ""


if __name__ == "__main__":
    test_new_book_best_model()
    test_new_book_tfidf_candidate()
    print("Semua uji coba rekomendasi penutup competency gap karyawan BERHASIL (100% PASS).")
