import { Router } from "express";
import { JobRepository, IJobRepository, AIJobRecord } from "./repositories/job.repository";
import { ModelOrchestrationService, QueueJobDTO } from "./services/orchestration.service";
import { OrchestrationController } from "./controllers/orchestration.controller";

export { IJobRepository, JobRepository, AIJobRecord, ModelOrchestrationService, QueueJobDTO };

export function createModelOrchestrationModule(): { router: Router; service: ModelOrchestrationService } {
  const repository = new JobRepository();
  const service = new ModelOrchestrationService(repository);
  const controller = new OrchestrationController(service);

  const router = Router();

  router.post("/jobs", controller.queueJob);
  router.get("/jobs/:jobId", controller.getJob);
  router.get("/jobs/match/:matchId", controller.getMatchJobs);

  return { router, service };
}
