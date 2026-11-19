import { ArrowUpRight } from "lucide-react";
import type { Book, OpenBook } from "../types";
import { Cover } from "./Cover";

export function BookCard({ book, open }: { book: Book; open: OpenBook }) {
  return (
    <article className="book-card">
      <button
        className="book-visual"
        onClick={() => open(book)}
        aria-label={`Lihat detail ${book.title}`}
      >
        <Cover book={book} />
        <span className="cover-hover">
          <ArrowUpRight size={20} />
        </span>
      </button>
      <div className="book-meta">
        <span className="book-category">{book.category}</span>
        <h3>
          <button onClick={() => open(book)}>{book.title}</button>
        </h3>
        <p>{book.author}</p>
        <div className="book-footer">
          <span>{book.year}</span>
          <button onClick={() => open(book)} aria-label={`Lihat detail ${book.title}`}>
            Lihat detail <ArrowUpRight size={14} />
          </button>
        </div>
      </div>
    </article>
  );
}
