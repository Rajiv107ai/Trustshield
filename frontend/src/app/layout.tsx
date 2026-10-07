import type { Metadata } from "next";
import "./globals.css";
import { ViewModeProvider } from "@/context/ViewModeContext";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { SearchModal } from "@/components/layout/SearchModal";

export const metadata: Metadata = {
  title: "TrustShield AI — Fraud Intelligence & Trust Engine",
  description:
    "Production enterprise e-commerce fraud intelligence platform combining relational GNNs, continuous-time dynamics, CLIP multimodal search, and conformal trust scoring.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full">
      <body className="min-h-full flex bg-[#0B0F14] text-[#E8EDF3] antialiased">
        <ViewModeProvider>
          <div className="flex w-full min-h-screen">
            {/* Sidebar */}
            <Sidebar />

            {/* Main Application Area */}
            <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">
              <Header />
              <main className="flex-1 p-6 max-w-7xl w-full mx-auto">{children}</main>
            </div>
          </div>

          {/* Quick Search Modal */}
          <SearchModal />
        </ViewModeProvider>
      </body>
    </html>
  );
}
