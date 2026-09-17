import { Router } from "express";
import {
  AnalyticsRepository,
  IAnalyticsRepository,
  TacticalFrameSnapshot,
  PitchControlGrid,
  PlayerTrackPoint,
} from "./repositories/analytics.repository";
import { MatchAnalyticsService } from "./services/analytics.service";
import { AnalyticsController } from "./controllers/analytics.controller";

export {
  IAnalyticsRepository,
  AnalyticsRepository,
  TacticalFrameSnapshot,
  PitchControlGrid,
  PlayerTrackPoint,
  MatchAnalyticsService,
};

export function createMatchAnalyticsModule(): { router: Router; service: MatchAnalyticsService } {
  const repository = new AnalyticsRepository();
  const service = new MatchAnalyticsService(repository);
  const controller = new AnalyticsController(service);

  const router = Router();

  router.get("/:matchId/tactical-frame", controller.getTacticalFrame);
  router.get("/:matchId/pitch-control", controller.getPitchControl);
  router.get("/:matchId/heatmap/:playerId", controller.getPlayerHeatmap);

  return { router, service };
}
