import { TacticalFrameSnapshot, MatchStatsSummary } from "../types/match";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:4000/api";

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function fetchMatchSummary(matchId: string): Promise<MatchStatsSummary> {
  const res = await fetch(`${API_BASE}/stats/${matchId}/summary`);
  if (!res.ok) throw new Error("Failed to fetch match summary");
  const json = await res.json();
  return json.data;
}

export async function fetchTacticalFrame(matchId: string, frame: number): Promise<TacticalFrameSnapshot> {
  const res = await fetch(`${API_BASE}/analytics/${matchId}/tactical-frame?frame=${frame}`);
  if (!res.ok) throw new Error("Failed to fetch tactical frame");
  const json = await res.json();
  return json.data;
}
