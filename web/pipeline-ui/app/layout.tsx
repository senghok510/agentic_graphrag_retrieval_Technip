import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Tender Pipeline — Q&A",
  description: "Ask tender/ITB questions and watch the Graph+Vector RAG pipeline run, stage by stage.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
