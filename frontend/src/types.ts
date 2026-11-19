export type Page = "dashboard" | "recommendations" | "mapping" | "catalog" | "settings";
export type Book = {
  id: string;
  title: string;
  author: string;
  year: number | string;
  score?: number;
  tags: string[];
  category: string;
};
export type OpenBook = (book: Book) => void;
