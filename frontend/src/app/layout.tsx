import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Dwarpal — Gate Capture Prototype",
  description: "Computer vision + OCR prototype for yard gate truck/trailer data capture",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <nav className="border-b border-neutral-200 px-8 py-3 flex gap-6 text-sm">
          <a href="/" className="font-semibold">Dwarpal</a>
          <a href="/" className="hover:underline">Upload</a>
          <a href="/review" className="hover:underline">Review queue</a>
          <a href="/dashboard" className="hover:underline">Dashboard</a>
          <a href="/feasibility" className="hover:underline">Feasibility</a>
        </nav>
        {children}
      </body>
    </html>
  );
}
