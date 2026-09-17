import { Router } from "express";
import {
  StatsRepository,
  IStatsRepository,
  MatchStatsSummary,
  TeamMatchStats,
  PlayerPerformanceStats,
} from "./repositories/stats.repository";
import { StatsAggregationService } from "./services/stats.service";
import { StatsController } from "./controllers/stats.controller";

export {
  IStatsRepository,
  StatsRepository,
  MatchStatsSummary,
  TeamMatchStats,
  PlayerPerformanceStats,
  StatsAggregationService,
};

export function createStatsAggregationModule(): { router: Router; service: StatsAggregationService } {
  const repository = new StatsRepository();
  const service = new StatsAggregationService(repository);
  const controller = new StatsController(service);

  const router = Router();

  router.get("/:matchId/summary", controller.getMatchSummary);
  router.get("/:matchId/players", controller.getPlayerStats);

  return { router, service };
}
