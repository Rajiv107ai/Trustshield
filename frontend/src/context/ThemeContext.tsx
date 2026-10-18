"use client";

import React, { createContext, useContext, useEffect, useState } from "react";

type Theme = "dark" | "light";

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

function applyThemeToDocument(t: Theme) {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  root.classList.remove("light", "dark");
  root.classList.add(t);
  root.setAttribute("data-theme", t);
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>("dark");

  useEffect(() => {
    let initialTheme: Theme = "dark";
    try {
      const stored = localStorage.getItem("trustshield_theme") as Theme | null;
      if (stored === "light" || stored === "dark") {
        initialTheme = stored;
      } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
        initialTheme = "light";
      }
    } catch {
      // ignore SSR or storage exceptions
    }
    applyThemeToDocument(initialTheme);
    queueMicrotask(() => {
      setThemeState(initialTheme);
    });
  }, []);

  const setTheme = (t: Theme) => {
    setThemeState(t);
    applyThemeToDocument(t);
    try {
      localStorage.setItem("trustshield_theme", t);
    } catch {
      // ignore
    }
  };

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
  };

  return (
    <ThemeContext.Provider value={{ theme, setTheme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}
