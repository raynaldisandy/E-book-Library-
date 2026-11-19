import { ArrowRight, ArrowUpRight, BookOpenText, ChevronDown, FileSearch, RefreshCw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { mapBook } from "../api";
import { Empty, Skeleton } from "../components/LibraryStates";
import { Heading } from "../components/PageHeading";
import { similarityDescription, similarityFormat } from "../data/similarity";

type Result = Awaited<ReturnType<typeof mapBook>>["recommendations"];

export function Mapping({ onMapped }: { onMapped: () => void }) {
  const [title, setTitle] = useState("");
  const [abstract, setAbstract] = useState("");
  const [touched, setTouched] = useState({ title: false, abstract: false });
  const [status, setStatus] = useState<"initial" | "loading" | "ready" | "error">("initial");
  const [result, setResult] = useState<Result>([]);
  const [error, setError] = useState("");
  const [analyzedTitle, setAnalyzedTitle] = useState("");
  const activeRequest = useRef<AbortController | null>(null);
  useEffect(() => () => activeRequest.current?.abort(), []);

  const titleError = title.trim().length < 2;
  const abstractError = abstract.trim().length < 20 || abstract.length > 2000;
  async function analyze() {
    setTouched({ title: true, abstract: true });
    if (titleError || abstractError || status === "loading") return;
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    setStatus("loading");
    try {
      const data = await mapBook(title.trim(), abstract.trim(), controller.signal);
      if (controller.signal.aborted) return;
      setAnalyzedTitle(data.query_title);
      setResult(data.recommendations);
      setStatus("ready");
      onMapped();
    } catch (error) {
      if (controller.signal.aborted) return;
      setError(error instanceof Error ? error.message : "Pemetaan belum dapat dimuat.");
      setStatus("error");
    }
  }
  return (
    <>
      <Heading eyebrow="DARI ISI BUKU KE KOMPETENSI" title="Pemetaan Buku Baru">
        Kenali arah pembelajaran sebuah buku melalui analisis judul dan ringkasannya.
      </Heading>
      <div className="mapping-workspace">
        <section className="paper manuscript">
          <div className="panel-top">
            <span className="step-label">01 / INFORMASI BUKU</span>
            <BookOpenText size={19} />
          </div>
          <h2>Mulai dari ceritanya.</h2>
          <p className="muted">Masukkan judul dan ringkasan untuk memetakan kompetensi.</p>
          <button
            className="preset"
            disabled={status === "loading"}
            onClick={() => {
              setTitle("Enterprise Risk Management");
              setAbstract(
                "Panduan manajemen risiko untuk mengidentifikasi risiko, mengukur dampak, menyusun mitigasi, serta membangun pengendalian dan pengambilan keputusan berbasis data.",
              );
              setTouched({ title: false, abstract: false });
              setStatus("initial");
            }}
          >
            Isi contoh: Enterprise Risk Management <ArrowUpRight size={14} />
          </button>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              analyze();
            }}
            noValidate
          >
            <div className="form-field">
              <label htmlFor="book-title">
                Judul buku <span>*</span>
              </label>
              <input
                id="book-title"
                value={title}
                disabled={status === "loading"}
                placeholder="Tuliskan judul buku"
                aria-invalid={touched.title && titleError}
                aria-describedby="title-help"
                onBlur={() => setTouched((t) => ({ ...t, title: true }))}
                onChange={(e) => {
                  setTitle(e.target.value);
                  setStatus("initial");
                }}
              />
              <p
                id="title-help"
                className={touched.title && titleError ? "field-error" : "field-hint"}
              >
                {touched.title && titleError
                  ? "Judul buku minimal 2 karakter."
                  : "Gunakan judul lengkap, minimal 2 karakter."}
              </p>
            </div>
            <div className="form-field">
              <label htmlFor="abstract">
                Abstrak atau ringkasan <span>*</span>
              </label>
              <textarea
                id="abstract"
                value={abstract}
                disabled={status === "loading"}
                placeholder="Ceritakan gagasan utama, topik, dan pokok pembahasan buku…"
                aria-invalid={touched.abstract && abstractError}
                aria-describedby="abstract-help abstract-count"
                onBlur={() => setTouched((t) => ({ ...t, abstract: true }))}
                onChange={(e) => {
                  setAbstract(e.target.value);
                  setStatus("initial");
                }}
              />
              <div className="field-bottom">
                <p
                  id="abstract-help"
                  className={touched.abstract && abstractError ? "field-error" : "field-hint"}
                >
                  {touched.abstract && abstractError
                    ? abstract.length > 2000
                      ? "Maksimal 2.000 karakter."
                      : "Ringkasan minimal 20 karakter."
                    : "20–2.000 karakter."}
                </p>
                <span
                  id="abstract-count"
                  className={abstract.length > 2000 ? "field-error" : "field-hint"}
                >
                  {abstract.length.toLocaleString("id-ID")} / 2.000
                </span>
              </div>
            </div>
            <button className="btn btn-primary full-width" disabled={status === "loading"}>
              <FileSearch size={17} />
              {status === "loading" ? "Menganalisis buku…" : "Analisis buku"}
              <ArrowRight size={16} />
            </button>
          </form>
        </section>
        <section className="mapping-results" aria-live="polite">
          <div className="panel-top">
            <span className="step-label">02 / PETA KOMPETENSI</span>
            <span className="muted">Maksimal 5 hasil</span>
          </div>
          <h2>
            {status === "ready" && result.length
              ? `${result.length} arah pengembangan`
              : "Temukan keterkaitannya."}
          </h2>
          <p className="muted">
            {status === "ready"
              ? `Hasil analisis untuk “${analyzedTitle}”`
              : "Kompetensi disusun berdasarkan kemiripan makna teks."}
          </p>
          {status === "loading" ? (
            <Skeleton />
          ) : status === "error" ? (
            <Empty title="Pemetaan belum dapat dimuat">
              <span role="alert">{error}</span>
              <button className="text-link" onClick={analyze}>
                <RefreshCw size={15} /> Coba lagi
              </button>
            </Empty>
          ) : status === "ready" ? (
            result.length ? (
              <div className="competency-results">
                {result.map(({ competency_id, competency_name, score }, i) => (
                  <details className="competency-result" key={competency_id}>
                    <summary>
                      <span className="competency-rank">{String(i + 1).padStart(2, "0")}</span>
                      <span>
                        <small>{competency_id}</small>
                        <strong>{competency_name}</strong>
                      </span>
                      <ChevronDown size={17} />
                      <span className="sr-only">Rincian skor kemiripan</span>
                    </summary>
                    <div className="competency-score">
                      <span>
                        Skor kemiripan <strong>{similarityFormat.format(score)}</strong>
                      </span>
                      <p>{similarityDescription}</p>
                    </div>
                  </details>
                ))}
                <p className="field-hint disclosure-help">
                  Buka setiap kompetensi untuk melihat skor kemiripan.
                </p>
              </div>
            ) : (
              <Empty icon="map" title="Belum ditemukan keterkaitan">
                Tambahkan konteks isi buku atau gunakan contoh yang tersedia.
              </Empty>
            )
          ) : (
            <Empty icon="map" title="Setiap buku punya arah belajar">
              Lengkapi judul dan ringkasan di sebelah kiri. Hasil pemetaan akan hadir di sini.
            </Empty>
          )}
        </section>
      </div>
    </>
  );
}
