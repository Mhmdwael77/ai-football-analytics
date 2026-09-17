export interface PitchCoordinate {
  x: number;
  y: number;
  vx?: number;
  vy?: number;
}

export interface PlayerTrackPoint {
  playerId: string;
  jerseyNumber: number;
  team: "home" | "away";
  position: PitchCoordinate;
  speedMps: number;
}

export interface TacticalFrameSnapshot {
  frameNumber: number;
  timestampSeconds: number;
  players: PlayerTrackPoint[];
  ball: PitchCoordinate & { z?: number };
}

export interface TeamMatchStats {
  team: "home" | "away";
  possessionPercent: number;
  totalPasses: number;
  successfulPasses: number;
  passAccuracyPercent: number;
  shotsOnTarget: number;
  expectedGoals: number;
  distanceCoveredKm: number;
}

export interface MatchStatsSummary {
  matchId: string;
  homeTeam: TeamMatchStats;
  awayTeam: TeamMatchStats;
  turnovers: number;
  intensityIndex: number;
  updatedAt: string;
}

export interface MatchEventItem {
  id: string;
  minute: number;
  second: number;
  type: "pass" | "shot" | "tackle" | "interception" | "foul";
  team: "home" | "away";
  player: string;
  detail: string;
  success: boolean;
}
