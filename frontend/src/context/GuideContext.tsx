"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { OnboardingGuideModal } from "@/components/common/OnboardingGuideModal";
import { CommandPalette } from "@/components/common/CommandPalette";

interface GuideContextValue {
  isTourOpen: boolean;
  openTour: () => void;
  closeTour: () => void;
  isPaletteOpen: boolean;
  openPalette: () => void;
  closePalette: () => void;
}

const GuideContext = createContext<GuideContextValue | undefined>(undefined);

export function GuideProvider({ children }: { children: React.ReactNode }) {
  const [isTourOpen, setIsTourOpen] = useState(false);
  const [isPaletteOpen, setIsPaletteOpen] = useState(false);

  // Global Ctrl+K / Cmd+K listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setIsPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const openTour = () => setIsTourOpen(true);
  const closeTour = () => setIsTourOpen(false);
  const openPalette = () => setIsPaletteOpen(true);
  const closePalette = () => setIsPaletteOpen(false);

  return (
    <GuideContext.Provider
      value={{
        isTourOpen,
        openTour,
        closeTour,
        isPaletteOpen,
        openPalette,
        closePalette,
      }}
    >
      {children}
      <OnboardingGuideModal isOpen={isTourOpen} onClose={closeTour} />
      <CommandPalette isOpen={isPaletteOpen} onClose={closePalette} onOpenTour={openTour} />
    </GuideContext.Provider>
  );
}

export function useGuide(): GuideContextValue {
  const ctx = useContext(GuideContext);
  if (!ctx) {
    return {
      isTourOpen: false,
      openTour: () => {},
      closeTour: () => {},
      isPaletteOpen: false,
      openPalette: () => {},
      closePalette: () => {},
    };
  }
  return ctx;
}
