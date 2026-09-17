import {
  IAnalyticsRepository,
  TacticalFrameSnapshot,
  PitchControlGrid,
} from "../repositories/analytics.repository";
import { eventBus } from "../../../shared/events/event-bus";
import { getTracer } from "../../../observability/tracer";
import { logger } from "../../../observability/logger";
import { NotFoundError } from "../../../shared/errors/app-error";

export class MatchAnalyticsService {
  constructor(private readonly repo: IAnalyticsRepository) {
    this.initEventListeners();
  }

  private initEventListeners(): void {
    eventBus.subscribe<{ matchId: string; jobId: string }>(
      "MATCH_ANALYSIS_COMPLETED",
      async (event) => {
        logger.info(
          `[MatchAnalytics] Processing completed analysis for match ${event.payload.matchId}`
        );
        // Pre-cache tactical analytics or trigger summary generation
      }
    );
  }

  public async getTacticalFrame(matchId: string, frameNumber: number): Promise<TacticalFrameSnapshot> {
    const tracer = getTracer("match-analytics");
    return tracer.startActiveSpan("MatchAnalyticsService.getTacticalFrame", async (span) => {
      try {
        span.setAttribute("match.id", matchId);
        span.setAttribute("frame.number", frameNumber);

        const frame = await this.repo.getFrame(matchId, frameNumber);
        if (!frame) {
          throw new NotFoundError("Frame", frameNumber);
        }
        return frame;
      } finally {
        span.end();
      }
    });
  }

  public async getPitchControl(matchId: string, frameNumber: number): Promise<PitchControlGrid> {
    const tracer = getTracer("match-analytics");
    return tracer.startActiveSpan("MatchAnalyticsService.getPitchControl", async (span) => {
      try {
        span.setAttribute("match.id", matchId);
        span.setAttribute("frame.number", frameNumber);
        return await this.repo.getPitchControl(matchId, frameNumber);
      } finally {
        span.end();
      }
    });
  }

  public async getPlayerHeatmap(matchId: string, playerId: string): Promise<{
    matchId: string;
    playerId: string;
    densityPoints: Array<{ x: number; y: number; value: number }>;
  }> {
    // Generate synthetic density points representing player heat dispersion
    const densityPoints: Array<{ x: number; y: number; value: number }> = [];
    for (let i = 0; i < 60; i++) {
      densityPoints.push({
        x: Math.round(30 + Math.random() * 40),
        y: Math.round(15 + Math.random() * 38),
        value: Number((0.3 + Math.random() * 0.7).toFixed(2)),
      });
    }

    return {
      matchId,
      playerId,
      densityPoints,
    };
  }
}
