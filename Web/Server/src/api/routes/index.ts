import { Router } from "express";
import { healthRouter } from "./health";
import { createVideoManagementModule } from "../../modules/video-management";
import { createModelOrchestrationModule } from "../../modules/model-orchestration";
import { createMatchAnalyticsModule } from "../../modules/match-analytics";
import { createStatsAggregationModule } from "../../modules/stats-aggregation";

export function createApiRouter(): Router {
  const apiRouter = Router();

  // Global health diagnostic routes
  apiRouter.use("/health", healthRouter);

  // Mount isolated modular monolith domains
  const videoModule = createVideoManagementModule();
  apiRouter.use("/videos", videoModule.router);

  const orchestrationModule = createModelOrchestrationModule();
  apiRouter.use("/orchestration", orchestrationModule.router);

  const analyticsModule = createMatchAnalyticsModule();
  apiRouter.use("/analytics", analyticsModule.router);

  const statsModule = createStatsAggregationModule();
  apiRouter.use("/stats", statsModule.router);

  return apiRouter;
}
