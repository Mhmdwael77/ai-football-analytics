import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { useMatchStore } from "../store/useMatchStore";
import { fetchTacticalFrame, fetchMatchSummary } from "../lib/api";
import { TacticalFrameSnapshot, MatchStatsSummary, MatchEventItem } from "../types/match";

const defaultFrame: TacticalFrameSnapshot = {
  frameNumber: 0,
  timestampSeconds: 0,
  players: [
    { playerId: "h_1", jerseyNumber: 1, team: "home", position: { x: 8, y: 34 }, speedMps: 0 },
    { playerId: "h_4", jerseyNumber: 4, team: "home", position: { x: 25, y: 26 }, speedMps: 2.1 },
    { playerId: "h_10", jerseyNumber: 10, team: "home", position: { x: 48, y: 34 }, speedMps: 4.5 },
    { playerId: "h_9", jerseyNumber: 9, team: "home", position: { x: 68, y: 34 }, speedMps: 3.8 },
    { playerId: "a_1", jerseyNumber: 1, team: "away", position: { x: 97, y: 34 }, speedMps: 0 },
    { playerId: "a_4", jerseyNumber: 4, team: "away", position: { x: 80, y: 26 }, speedMps: 1.8 },
    { playerId: "a_8", jerseyNumber: 8, team: "away", position: { x: 58, y: 30 }, speedMps: 3.2 },
  ],
  ball: { x: 50, y: 34, z: 0.1 },
};

const defaultStats: MatchStatsSummary = {
  matchId: "demo",
  homeTeam: {
    team: "home",
    possessionPercent: 57,
    totalPasses: 412,
    successfulPasses: 350,
    passAccuracyPercent: 85,
    shotsOnTarget: 6,
    expectedGoals: 1.72,
    distanceCoveredKm: 104.2,
  },
  awayTeam: {
    team: "away",
    possessionPercent: 43,
    totalPasses: 310,
    successfulPasses: 248,
    passAccuracyPercent: 80,
    shotsOnTarget: 3,
    expectedGoals: 0.88,
    distanceCoveredKm: 106.8,
  },
  turnovers: 24,
  intensityIndex: 76,
  updatedAt: new Date().toISOString(),
};

const mockEvents: MatchEventItem[] = [
  { id: "e1", minute: 64, second: 12, type: "pass", team: "home", player: "#10 Playmaker", detail: "Progressive ground pass to final third", success: true },
  { id: "e2", minute: 63, second: 45, type: "tackle", team: "away", player: "#4 Centerback", detail: "Sliding interception on right wing", success: true },
  { id: "e3", minute: 61, second: 20, type: "shot", team: "home", player: "#9 Striker", detail: "Curled shot from edge of box (xG: 0.28)", success: true },
  { id: "e4", minute: 58, second: 0, type: "pass", team: "away", player: "#8 Midfielder", detail: "Turnover caused under high press", success: false },
];

export function useMatchTelemetry() {
  const { currentMatchId, currentFrame, isPlaying, incrementFrame } = useMatchStore();

  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(() => {
      incrementFrame();
    }, 120);
    return () => clearInterval(timer);
  }, [isPlaying, incrementFrame]);

  const frameQuery = useQuery({
    queryKey: ["tactical-frame", currentMatchId, currentFrame],
    queryFn: () => fetchTacticalFrame(currentMatchId, currentFrame),
    placeholderData: defaultFrame,
  });

  const statsQuery = useQuery({
    queryKey: ["match-summary", currentMatchId],
    queryFn: () => fetchMatchSummary(currentMatchId),
    placeholderData: defaultStats,
  });

  return {
    frame: frameQuery.data || defaultFrame,
    stats: statsQuery.data || defaultStats,
    events: mockEvents,
    isLoading: frameQuery.isLoading && statsQuery.isLoading,
  };
}
