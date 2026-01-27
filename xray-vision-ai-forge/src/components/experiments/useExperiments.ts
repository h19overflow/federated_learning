import { useState, useEffect, useCallback } from "react";
import { toast } from "sonner";
import api from "@/services/api";
import { RunSummary } from "@/types/runs";

export type ViewMode = "detailed" | "concise" | "compact";

export const useExperiments = () => {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("experiments");
  const [viewMode, setViewMode] = useState<ViewMode>(() => {
    const saved = localStorage.getItem("experiments-view-mode");
    return (saved as ViewMode) || "detailed";
  });

  useEffect(() => {
    localStorage.setItem("experiments-view-mode", viewMode);
  }, [viewMode]);

  useEffect(() => {
    const fetchRuns = async () => {
      try {
        setLoading(true);
        const response = await api.results.listRuns();
        setRuns(response.runs);
      } catch (err) {
        console.error("Error fetching runs:", err);
        const message = err instanceof Error ? err.message : "Failed to load saved experiments";
        setError(message);
        toast.error("Failed to load experiments");
      } finally {
        setLoading(false);
      }
    };

    fetchRuns();
  }, []);

  const refresh = useCallback(() => {
    window.location.reload();
  }, []);

  return {
    runs,
    loading,
    error,
    activeTab,
    setActiveTab,
    viewMode,
    setViewMode,
    refresh,
  };
};
