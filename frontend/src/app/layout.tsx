import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "@/context/ThemeContext";
import { ViewModeProvider } from "@/context/ViewModeContext";
import { GuideProvider } from "@/context/GuideContext";
import { AppShell } from "@/components/layout/AppShell";

export const metadata: Metadata = {
  title: "TrustShield AI — Fraud Intelligence & Trust Engine",
  description:
    "Production enterprise e-commerce fraud intelligence platform combining point-in-time graph features, TreeSHAP attributions, and conformal uncertainty guarantees.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full" suppressHydrationWarning>
      <body className="min-h-full flex antialiased bg-[var(--background)] text-[var(--text-primary)]">
        <ThemeProvider>
          <ViewModeProvider>
            <GuideProvider>
              <AppShell>{children}</AppShell>
            </GuideProvider>
          </ViewModeProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
