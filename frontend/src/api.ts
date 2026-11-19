import type { Book } from "./types";
import { catalogGroups, type CatalogGroup } from "./data/categories";

export type Competency = { competency_id: string; competency_name: string };

export type Health = {
  status: string;
  version: string;
  active_model: string;
  model_type: string;
  is_fallback: boolean;
  catalog_books_count: number;
  competencies_count: number;
  embedding_dimension?: number | null;
  catalog_exported_at?: string | null;
};

type CatalogBook = {
  book_id: string;
  judul: string;
  author: string;
  tahun: string | number;
  kategori_buku?: string;
  kategori?: string;
};

type ScoredBook = CatalogBook & { rank: number; score: number; model: string; is_fallback: boolean };

export type RecommendationsResult = { competency_name: string; competency_id: string; books: ScoredBook[] };
export type CatalogSort = "newest" | "oldest" | "title";
export const catalogPageSize = 4;

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === "string" ? body.detail : "Layanan tidak dapat diakses. Coba lagi.");
  }
  return response.json();
}

function post<T>(endpoint: string, payload: object, signal?: AbortSignal) {
  return request<T>(`/api/v1/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
}

export function getCompetencies(signal?: AbortSignal) {
  return request<{ competencies: Competency[] }>("/api/v1/competencies", { signal });
}

export function getHealth(signal?: AbortSignal) {
  return request<Health>("/health", { signal, cache: "no-store" });
}

export function getRecommendations(competencyId: string, category: CatalogGroup | "", signal?: AbortSignal) {
  return post<RecommendationsResult>("recommend-books", {
    competency_id: competencyId,
    top_k: 20,
    categories: category ? catalogGroups[category] : undefined,
  }, signal);
}

export function mapBook(title: string, abstract: string, signal?: AbortSignal) {
  return post<{ query_title: string; recommendations: (Competency & { score: number; model: string; is_fallback: boolean })[] }>(
    "recommend-competencies",
    { title, abstract, top_k: 5 },
    signal,
  );
}

export function getBookDetail(id: string, signal: AbortSignal) {
  return request<{ book_id: string; abstrak: string }>(`/api/v1/catalog/${encodeURIComponent(id)}`, { signal });
}

export function searchCatalog(term: string, category: CatalogGroup | "Semua", offset: number, sort: CatalogSort, signal: AbortSignal) {
  const params = new URLSearchParams({ q: term, limit: String(catalogPageSize), offset: String(offset), sort });
  if (category !== "Semua") {
    for (const value of catalogGroups[category]) params.append("category", value);
  }
  return request<{ books: CatalogBook[]; total_matched: number }>(`/api/v1/catalog?${params}`, { signal });
}

export function fromCatalog(book: CatalogBook): Book {
  return {
    id: book.book_id,
    title: book.judul,
    author: book.author,
    year: book.tahun,
    category: book.kategori_buku ?? book.kategori ?? "",
    tags: [],
  };
}

export function fromRecommendation(book: ScoredBook, competency: string): Book {
  return { ...fromCatalog(book), score: book.score, tags: [competency] };
}

export function findCompetency(competencies: Competency[], query: string) {
  const term = query.trim().toLowerCase();
  return competencies.find(({ competency_id, competency_name }) =>
    [competency_name, competency_id, `${competency_name} (${competency_id})`]
      .some((value) => value.toLowerCase() === term),
  );
}
