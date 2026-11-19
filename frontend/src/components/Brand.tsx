import { BookOpen } from "lucide-react";

export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className="brand">
      <div className="brand-icon"><BookOpen size={26} strokeWidth={1.5} /></div>
      {!compact && <div><strong>e-Library<span>BPJS Kesehatan</span></strong></div>}
    </div>
  );
}
