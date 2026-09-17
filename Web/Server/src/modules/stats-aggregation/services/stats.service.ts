import {
  IStatsRepository,
  MatchStatsSummary,
  PlayerPerformanceStats,
} from "../repositories/stats.repository";
import { eventBus } from "../../../shared/events/event-bus";
import { getTracer } from "../../../observability/tracer";
import { logger } from "../../../observability/logger";
import { NotFoundError } from "../../../shared/errors/app-error";

export class StatsAggregationService {
  constructor(private readonly repo: IStatsRepository) {
    this.initEventListeners();
  }

  private initEventListeners(): void {
    eventBus.subscribe<{ matchId: string; jobId: string }>(
      "MATCH_ANALYSIS_COMPLETED",
      async (event) => {
        logger.info(
          `[StatsAggregation] Re-aggregating team and player statistics for match ${event.payload.matchId}`
        );
        // Recalculate metrics when new AI pipeline detection finishes
      }
    );
  }

  public async getMatchSummary(matchId: string): Promise<MatchStatsSummary> {
    const tracer = getTracer("stats-aggregation");
    return tracer.startActiveSpan("StatsAggregationService.getMatchSummary", async (span) => {
      try {
        span.setAttribute("match.id", matchId);
        const summary = await this.repo.getMatchSummary(matchId);
        if (!summary) {
          throw new NotFoundError("Match Stats", matchId);
        }
        return summary;
      } finally {
        span.end();
      }
    });
  }

  public async getPlayerStats(matchId: string): Promise<PlayerPerformanceStats[]> {
    const tracer = getTracer("stats-aggregation");
    return tracer.startActiveSpan("StatsAggregationService.getPlayerStats", async (span) => {
      try {
        span.setAttribute("match.id", matchId);
        return await this.repo.getPlayerStats(matchId);
      } finally {
        span.end();
      }
    });
  }
}
