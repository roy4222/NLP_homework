import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = {
  title: "Chinese Legal Issue Triage",
  description: "NLP final project demo for Chinese legal issue classification.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
