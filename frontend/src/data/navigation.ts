import { FileSearch, LayoutGrid, Library, Settings, Sparkles } from "lucide-react";

export const nav = [
  { id: "dashboard", label: "Dashboard", icon: LayoutGrid },
  { id: "recommendations", label: "Rekomendasi Buku", icon: Sparkles },
  { id: "mapping", label: "Pemetaan Buku Baru", icon: FileSearch },
  { id: "catalog", label: "Katalog Buku", icon: Library },
  { id: "settings", label: "Pengaturan", icon: Settings },
] as const;
