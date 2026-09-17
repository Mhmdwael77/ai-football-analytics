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

export interface PlayerPerformanceStats {
  playerId: string;
  name: string;
  team: "home" | "away";
  jerseyNumber: number;
  distanceCoveredKm: number;
  topSpeedKmh: number;
  passesCompleted: number;
  passesAttempted: number;
  passAccuracyPercent: number;
  interceptions: number;
  minutesPlayed: number;
}

export interface MatchStatsSummary {
  matchId: string;
  homeTeam: TeamMatchStats;
  awayTeam: TeamMatchStats;
  turnovers: number;
  intensityIndex: number; // 0 - 100
  updatedAt: string;
}

export interface IStatsRepository {
  getMatchSummary(matchId: string): Promise<MatchStatsSummary | null>;
  saveMatchSummary(summary: MatchStatsSummary): Promise<void>;
  getPlayerStats(matchId: string): Promise<PlayerPerformanceStats[]>;
}

export class StatsRepository implements IStatsRepository {
  private summaries = new Map<string, MatchStatsSummary>();

  public async getMatchSummary(matchId: string): Promise<MatchStatsSummary | null> {
    const existing = this.summaries.get(matchId);
    if (existing) return existing;
    return this.generateDefaultSummary(matchId);
  }

  public async saveMatchSummary(summary: MatchStatsSummary): Promise<void> {
    this.summaries.set(summary.matchId, summary);
  }

  public async getPlayerStats(_matchId: string): Promise<PlayerPerformanceStats[]> {
    return [
      {
        playerId: "h_10",
        name: "Playmaker",
        team: "home",
        jerseyNumber: 10,
        distanceCoveredKm: 9.8,
        topSpeedKmh: 31.4,
        passesCompleted: 54,
        passesAttempted: 62,
        passAccuracyPercent: 87.1,
        interceptions: 4,
        minutesPlayed: 90,
      },
      {
        playerId: "h_9",
        name: "Striker",
        team: "home",
        jerseyNumber: 9,
        distanceCoveredKm: 8.4,
        topSpeedKmh: 33.8,
        passesCompleted: 18,
        passesAttempted: 24,
        passAccuracyPercent: 75.0,
        interceptions: 1,
        minutesPlayed: 90,
      },
      {
        playerId: "a_8",
        name: "Midfielder",
        team: "away",
        jerseyNumber: 8,
        distanceCoveredKm: 10.2,
        topSpeedKmh: 29.7,
        passesCompleted: 48,
        passesAttempted: 55,
        passAccuracyPercent: 87.2,
        interceptions: 6,
        minutesPlayed: 90,
      },
    ];
  }

  private generateDefaultSummary(matchId: string): MatchStatsSummary {
    return {
      matchId,
      homeTeam: {
        team: "home",
        possessionPercent: 58.4,
        totalPasses: 492,
        successfulPasses: 421,
        passAccuracyPercent: 85.6,
        shotsOnTarget: 7,
        expectedGoals: 1.84,
        distanceCoveredKm: 108.2,
      },
      awayTeam: {
        team: "away",
        possessionPercent: 41.6,
        totalPasses: 345,
        successfulPasses: 279,
        passAccuracyPercent: 80.9,
        shotsOnTarget: 3,
        expectedGoals: 0.92,
        distanceCoveredKm: 111.4,
      },
      turnovers: 28,
      intensityIndex: 78,
      updatedAt: new Date().toISOString(),
    };
  }
}
