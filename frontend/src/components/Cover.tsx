import type { CSSProperties } from "react";
import type { Book } from "../types";

const palettes = [
  ["#cddbbd", "#213e31", "#74986c"],
  ["#e9b75a", "#45331b", "#b67d28"],
  ["#b7b8cf", "#302e48", "#757491"],
  ["#c46d4c", "#fff5db", "#e4ac79"],
  ["#24736c", "#fff1d8", "#9dbbb0"],
  ["#e3d7bc", "#30494b", "#8b9b91"],
];

export function Cover({ book, className = "", variant }: { book: Book; className?: string; variant?: number }) {
  const number = Number(book.id.match(/\d+$/)?.[0] ?? "1");
  const edition = variant ?? ((Math.max(1, number) - 1) % palettes.length) + 1;
  const [color, ink, accent] = palettes[edition - 1];
  return (
    <div
      aria-hidden="true"
      className={`book-cover cover-${edition} ${className}`}
      style={
        {
          "--cover": color,
          "--cover-ink": ink,
          "--cover-accent": accent,
        } as CSSProperties
      }
    >
      <div className="cover-top">
        <span>PUSTAKA KOMPETENSI</span>
        <span>{String(number).padStart(2, "0")}</span>
      </div>
      <div className="cover-title"><span>{book.title}</span></div>
      <div className="cover-art">
        <svg viewBox="0 0 200 150" fill="none" preserveAspectRatio="xMidYMid meet">
          {edition % 3 === 1 ? (
            <>
              <path d="M20 150V75a80 80 0 0 1 160 0v75M40 150V75a60 60 0 0 1 120 0v75M60 150V75a40 40 0 0 1 80 0v75M80 150V75a20 20 0 0 1 40 0v75" />
              <path d="M0 115h200M0 135h200" />
            </>
          ) : edition % 3 === 2 ? (
            <>
              {[0, 1, 2, 3, 4].map((i) => (
                <g key={i} transform={`translate(${i * 28 - 15},${i * -11 + 30})`}>
                  <path d="M15 110L75 10l60 100Z" />
                  <path d="M15 110l60-35 60 35M75 10v65" />
                </g>
              ))}
            </>
          ) : (
            <>
              {[0, 1, 2, 3, 4].map((i) => (
                <ellipse
                  key={i}
                  cx="100"
                  cy="75"
                  rx={25 + i * 15}
                  ry="65"
                  transform={`rotate(${i * 30} 100 75)`}
                />
              ))}
              <circle cx="100" cy="75" r="5" fill="currentColor" />
            </>
          )}
        </svg>
      </div>
      <div className="cover-bottom">
        <span>{book.author}</span>
        <span>{book.year}</span>
      </div>
    </div>
  );
}
