import { AlertCircle, ArrowRight, ArrowUpRight, Check, ChevronDown, CircleHelp, RefreshCw, Search, Sparkles, X } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import { fromRecommendation, getRecommendations, type Competency, type RecommendationsResult } from "../api";
import { Cover } from "../components/Cover";
import { GroupFilters } from "../components/GroupFilters";
import { Empty, Skeleton } from "../components/LibraryStates";
import { Heading } from "../components/PageHeading";
import type { CatalogGroup } from "../data/categories";
import { similarityDescription } from "../data/similarity";
import type { OpenBook } from "../types";

const normalize = (value: string) => value.normalize("NFD").replace(/\p{Diacritic}/gu, "").toLowerCase().trim();

function CompetencyCombobox({
  competencies,
  value,
  onChange,
}: {
  competencies: Competency[];
  value: Competency | null;
  onChange: (value: Competency | null) => void;
}) {
  const id = useId();
  const [query, setQuery] = useState(value ? `${value.competency_id} · ${value.competency_name}` : "");
  const [expanded, setExpanded] = useState(false);
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const options = competencies.filter((c) =>
    normalize(`${c.competency_id} ${c.competency_name}`).includes(normalize(value && query === `${value.competency_id} · ${value.competency_name}` ? "" : query)),
  );
  useEffect(() => {
    if (value) setQuery(`${value.competency_id} · ${value.competency_name}`);
  }, [value]);
  useEffect(() => {
    if (expanded && options[active]) {
      document
        .getElementById(`${id}-${options[active].competency_id}`)
        ?.scrollIntoView({ block: "nearest" });
    }
  }, [active, expanded, id, query]);
  useEffect(() => {
    function outside(e: PointerEvent) {
      if (!root.current?.contains(e.target as Node)) setExpanded(false);
    }
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, []);
  function choose(c: Competency) {
    onChange(c);
    setQuery(`${c.competency_id} · ${c.competency_name}`);
    setExpanded(false);
  }
  return (
    <div ref={root} className="combobox">
      <label htmlFor={id}>Kompetensi yang ingin dikembangkan</label>
      <div className="input-with-icon">
        <Search size={18} />
        <input
          id={id}
          role="combobox"
          autoComplete="off"
          aria-autocomplete="list"
          aria-expanded={expanded}
          aria-controls={expanded ? `${id}-list` : undefined}
          aria-activedescendant={
            expanded && options[active] ? `${id}-${options[active].competency_id}` : undefined
          }
          placeholder="Cari kode atau nama kompetensi…"
          value={query}
          onFocus={() => {
            setExpanded(true);
            setActive(0);
          }}
          onBlur={() => {
            setExpanded(false);
            if (value) setQuery(`${value.competency_id} · ${value.competency_name}`);
          }}
          onChange={(e) => {
            setQuery(e.target.value);
            if (value) onChange(null);
            setExpanded(true);
            setActive(0);
          }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown" || e.key === "ArrowUp") {
              e.preventDefault();
              setExpanded(true);
              setActive((i) =>
                Math.max(0, Math.min(options.length - 1, i + (e.key === "ArrowDown" ? 1 : -1))),
              );
            } else if (e.key === "Enter" && expanded) {
              e.preventDefault();
              if (options[active]) choose(options[active]);
            } else if (e.key === "Escape") {
              setExpanded(false);
              if (value) setQuery(`${value.competency_id} · ${value.competency_name}`);
            }
          }}
        />
        <button
          type="button"
          aria-label={value || query ? "Hapus kompetensi" : "Buka pilihan kompetensi"}
          onClick={() => {
            if (value || query) {
              onChange(null);
              setQuery("");
            }
            root.current?.querySelector("input")?.focus();
            setActive(0);
            setExpanded(!expanded);
          }}
        >
          {value || query ? <X size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>
      {expanded && (
        <div
          className="combo-options"
          id={`${id}-list`}
          role="listbox"
          aria-label="Daftar kompetensi"
        >
          {options.length ? (
            options.map((c, i) => (
              <div
                role="option"
                aria-selected={value?.competency_id === c.competency_id}
                key={c.competency_id}
                id={`${id}-${c.competency_id}`}
                className={active === i ? "highlighted" : ""}
                onMouseEnter={() => setActive(i)}
                onPointerDown={(e) => e.preventDefault()}
                onClick={() => choose(c)}
              >
                <span>{c.competency_id}</span>
                {c.competency_name}
                {value?.competency_id === c.competency_id && <Check size={15} />}
              </div>
            ))
          ) : (
            <p className="combo-empty">Kompetensi tidak ditemukan. Coba kode atau nama lain.</p>
          )}
          <div className="combo-hint">
            {competencies.length} kompetensi tersedia
          </div>
        </div>
      )}
    </div>
  );
}

type RecommendationRequest = { competency: Competency; group: CatalogGroup | "Semua" };
type Outcome = { result?: RecommendationsResult; error?: string };

export function Recommendations({
  open, competencies, initialCompetency, competenciesError, retryCompetencies,
}: {
  open: OpenBook;
  competencies: Competency[];
  initialCompetency: Competency | null;
  competenciesError: boolean;
  retryCompetencies: () => void;
}) {
  const [competency, setCompetency] = useState(initialCompetency);
  const [group, setGroup] = useState<CatalogGroup | "Semua">("Semua");
  const [request, setRequest] = useState<RecommendationRequest | null>(
    initialCompetency ? { competency: initialCompetency, group: "Semua" } : null,
  );
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [query, setQuery] = useState("");
  const [validation, setValidation] = useState(false);

  useEffect(() => {
    if (!request) return;
    const controller = new AbortController();
    getRecommendations(request.competency.competency_id, request.group === "Semua" ? "" : request.group, controller.signal)
      .then((result) => { if (!controller.signal.aborted) setOutcome({ result }); })
      .catch((error: Error) => {
        if (!controller.signal.aborted) setOutcome({ error: error.message });
      });
    return () => controller.abort();
  }, [request]);

  function run() {
    if (!competency) {
      setValidation(true);
      return;
    }
    setValidation(false);
    setOutcome(null);
    setQuery("");
    setRequest({ competency, group });
  }

  const status = !request ? "initial" : outcome?.error ? "error" : outcome?.result ? "ready" : "loading";
  const searched = outcome?.result?.competency_name ?? "";
  const result = outcome?.result?.books ?? [];
  const filtered = result.filter(({ judul }) => normalize(judul).includes(normalize(query)))
    .map((book) => ({ book: fromRecommendation(book, searched), rank: book.rank }));
  const examples = ["C014", "C019", "C031"].flatMap((code) =>
    competencies.filter((item) => item.competency_id === code),
  );
  return (
    <>
      <Heading eyebrow="TEMUKAN PERSPEKTIF BARU" title="Rekomendasi Buku">
        Mulai dari kompetensi yang ingin Anda kembangkan. Temukan hingga lima bacaan yang relevan.
      </Heading>
      <form className="recommend-form paper" onSubmit={(event) => { event.preventDefault(); run(); }}>
        <div className="panel-top">
          <span className="step-label">01 / TENTUKAN KOMPETENSI</span>
        </div>
        <div className="recommend-controls">
          <div>
            <CompetencyCombobox
              competencies={competencies}
              value={competency}
              onChange={(c) => {
                setCompetency(c);
                setValidation(false);
                setRequest(null);
              }}
            />
            {validation && (
              <p className="field-error" role="alert">
                Pilih kompetensi dari daftar terlebih dahulu.
              </p>
            )}
          </div>
          <button className="btn btn-primary" type="submit" disabled={status === "loading" || !competencies.length}>
            <Sparkles size={17} />
            {status === "loading" ? "Mencari bacaan…" : "Cari rekomendasi"}
            <ArrowRight size={16} />
          </button>
        </div>
        <div className="examples">
          {examples.length > 0 && <span>Coba kompetensi:</span>}
          {examples.map((example) => (
            <button
              type="button"
              key={example.competency_id}
              onClick={() => {
                setCompetency(example);
                setRequest(null);
                setValidation(false);
              }}
            >
              {example.competency_name}
              <ArrowUpRight size={12} />
            </button>
          ))}
          {competenciesError && (
            <button type="button" className="text-link" onClick={retryCompetencies}>
              <RefreshCw size={15} /> Coba lagi memuat kompetensi
            </button>
          )}
        </div>
        <div className="recommend-bottom">
          <GroupFilters
            value={group}
            onChange={(g) => {
              setGroup(g);
              if (request && competency) {
                setOutcome(null);
                setQuery("");
                setRequest({ competency, group: g });
              }
            }}
            all="Semua kategori"
          />
        </div>
      </form>
      <section className="recommend-output" aria-live="polite">
        <div className="section-heading">
          <div>
            <div className="eyebrow">02 / PILIH BACAAN</div>
            <h2>
              {status === "ready"
                ? `${result.length ? `Top ${Math.min(5, result.length)}` : "Hasil"} rekomendasi`
                : "Rak rekomendasi Anda"}
            </h2>
            <p>
              {status === "ready"
                ? `Kompetensi: ${searched}`
                : "Bacaan terarah, dimulai dari kebutuhan Anda."}
            </p>
          </div>
          {status === "ready" && (
            <div className="input-with-icon result-search">
              <Search size={16} />
              <input
                aria-label="Saring judul hasil"
                placeholder="Saring judul hasil…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
          )}
        </div>
        {status === "initial" ? (
          <Empty icon="book" title="Bacaan yang relevan dimulai di sini">
            Pilih kompetensi atau gunakan salah satu contoh, lalu klik “Cari rekomendasi”.
          </Empty>
        ) : status === "loading" ? (
          <Skeleton />
        ) : status === "error" ? (
          <div className="empty-state error-state" role="alert">
            <AlertCircle size={32} />
            <h3>Rekomendasi belum dapat dimuat</h3>
            <p>{outcome?.error}</p>
            <button className="btn btn-primary" onClick={run}>
              <RefreshCw size={16} />
              Coba lagi
            </button>
          </div>
        ) : filtered.length ? (
          <div className="ranked-books">
            {filtered.slice(0, 5).map(({ book, rank }) => (
              <article className="ranked-book" key={book.id}>
                <span className="rank-number">{String(rank).padStart(2, "0")}</span>
                <button
                  className="rank-cover"
                  onClick={() => open(book)}
                  aria-label={`Lihat detail ${book.title}`}
                >
                  <Cover book={book} />
                </button>
                <div className="rank-content">
                  <span className="book-category">{book.category}</span>
                  <h3>{book.title}</h3>
                  <p>
                    {book.author} <span>·</span> {book.year}
                  </p>
                </div>
                <button className="text-link" onClick={() => open(book)}>
                  Lihat detail <ArrowUpRight size={16} />
                </button>
              </article>
            ))}
          </div>
        ) : (
          <Empty title="Belum ada buku yang cocok">
            {query
              ? "Tidak ada judul yang sesuai dengan kata kunci Anda."
              : "Kombinasi kompetensi dan kategori ini belum memiliki buku dalam katalog."}
            {query && (
              <button className="text-link" onClick={() => setQuery("")}>
                Hapus kata kunci <ArrowRight size={15} />
              </button>
            )}
          </Empty>
        )}
      </section>
      <div className="library-note">
        <CircleHelp size={15} />
        <p>
          {similarityDescription} Lihat detail buku untuk membaca skornya.
        </p>
      </div>
    </>
  );
}
