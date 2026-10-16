"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, useSyncExternalStore } from "react";
import { setBaseApiUrl, TrustShieldApi, isDemoMode, setDemoMode, getSessionApiKey, setSessionApiKey } from "@/lib/api/client";
import { ReadyResponse } from "@/lib/types/api";

const emptySubscribe = () => () => {};

const subscribeDemoMode = (callback: () => void) => {
  if (typeof window === "undefined") return () => {};
  window.addEventListener("trustshield:mode-change", callback);
  return () => window.removeEventListener("trustshield:mode-change", callback);
};

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
  isMounted: boolean;
  isDemo: boolean;
  setDemo: (enabled: boolean) => void;
  sessionApiKey: string;
  updateSessionApiKey: (key: string) => void;
}

const ViewModeContext = createContext<ViewModeContextType | undefined>(undefined);

export function ViewModeProvider({ children }: { children: React.ReactNode }) {
  const isMounted = useSyncExternalStore(emptySubscribe, () => true, () => false);
  const isDemo = useSyncExternalStore(subscribeDemoMode, isDemoMode, () => false);
  const [viewMode, setViewModeState] = useState<ViewMode>("executive");
  const [apiUrl, setApiUrlState] = useState<string>(
    (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "")
  );
  const [isLive, setIsLive] = useState<boolean>(false);
  const [sessionApiKey, setSessionApiKeyState] = useState<string>(() => getSessionApiKey());
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
    const timer = setTimeout(() => {
      try {
        const saved = localStorage.getItem("trustshield_view_mode");
        if (saved === "executive" || saved === "inspector") {
          setViewModeState(saved as ViewMode);
        }
      } catch {}

      try {
        const customUrl = localStorage.getItem("trustshield_api_url");
        if (customUrl) {
          setApiUrlState(customUrl.replace(/\/+$/, ""));
        }
      } catch {}
    }, 0);

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
      clearTimeout(timer);
      mounted = false;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  const setDemo = (enabled: boolean) => {
    setDemoMode(enabled);
    refreshHealth();
  };

  const updateSessionApiKey = (key: string) => {
    setSessionApiKey(key);
    setSessionApiKeyState(key.trim());
    refreshHealth();
  };

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
        isMounted,
        isDemo,
        setDemo,
        sessionApiKey,
        updateSessionApiKey,
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
