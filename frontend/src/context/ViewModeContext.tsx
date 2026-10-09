"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import { getBaseApiUrl, setBaseApiUrl, TrustShieldApi } from "@/lib/api/client";
import { ReadyResponse } from "@/lib/types/api";

export type ViewMode = "executive" | "inspector";

interface ViewModeContextType {
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  toggleViewMode: () => void;
  apiUrl: string;
  updateApiUrl: (url: string) => void;
  isLive: boolean;
  readyInfo: ReadyResponse | null;
  refreshHealth: () => Promise<void>;
  searchOpen: boolean;
  setSearchOpen: (open: boolean) => void;
}

const ViewModeContext = createContext<ViewModeContextType | undefined>(undefined);

export function ViewModeProvider({ children }: { children: React.ReactNode }) {
  // Initialize with deterministic defaults to prevent SSR/client hydration mismatch
  const [viewMode, setViewModeState] = useState<ViewMode>("executive");
  const [apiUrl, setApiUrlState] = useState<string>(() =>
    (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "")
  );
  const [isLive, setIsLive] = useState<boolean>(false);
  const [readyInfo, setReadyInfo] = useState<ReadyResponse | null>(null);
  const [searchOpen, setSearchOpen] = useState<boolean>(false);

  const refreshHealth = useCallback(async () => {
    try {
      const readyRes = await TrustShieldApi.getReady();
      setIsLive(readyRes.isLive);
      setReadyInfo(readyRes.data);
    } catch {
      setIsLive(false);
    }
  }, []);

  useEffect(() => {
    // Read persisted view mode & API URL on client mount after hydration has completed
    try {
      const savedMode = localStorage.getItem("trustshield_view_mode") as ViewMode | null;
      if (savedMode === "executive" || savedMode === "inspector") {
        setViewModeState(savedMode);
      }
    } catch {
      // Ignore localStorage access errors (e.g. sandboxed iframe or private browsing)
    }

    try {
      setApiUrlState(getBaseApiUrl());
    } catch {
      // Ignore
    }

    let mounted = true;
    TrustShieldApi.getReady()
      .then((readyRes) => {
        if (mounted) {
          setIsLive(readyRes.isLive);
          setReadyInfo(readyRes.data);
        }
      })
      .catch(() => {
        if (mounted) setIsLive(false);
      });

    // Hotkey handler for ⌘K / Ctrl+K
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setSearchOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      mounted = false;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  const setViewMode = (mode: ViewMode) => {
    setViewModeState(mode);
    try {
      localStorage.setItem("trustshield_view_mode", mode);
    } catch {
      // Ignore
    }
  };

  const toggleViewMode = () => {
    const next = viewMode === "executive" ? "inspector" : "executive";
    setViewMode(next);
  };

  const updateApiUrl = (url: string) => {
    setBaseApiUrl(url);
    setApiUrlState(url);
    refreshHealth();
  };

  return (
    <ViewModeContext.Provider
      value={{
        viewMode,
        setViewMode,
        toggleViewMode,
        apiUrl,
        updateApiUrl,
        isLive,
        readyInfo,
        refreshHealth,
        searchOpen,
        setSearchOpen,
      }}
    >
      {children}
    </ViewModeContext.Provider>
  );
}

export function useViewMode() {
  const context = useContext(ViewModeContext);
  if (!context) {
    throw new Error("useViewMode must be used within a ViewModeProvider");
  }
  return context;
}
