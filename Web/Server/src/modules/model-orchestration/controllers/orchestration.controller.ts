import { Request, Response, NextFunction } from "express";
import { z } from "zod";
import { ModelOrchestrationService } from "../services/orchestration.service";
import { ValidationError } from "../../../shared/errors/app-error";

const queueJobSchema = z.object({
  matchId: z.string().min(1, "matchId is required"),
  videoId: z.string().min(1, "videoId is required"),
  pipelineType: z
    .enum(["detection", "tracking", "pitch_calibration", "full_match_analysis"])
    .optional(),
});

export class OrchestrationController {
  constructor(private readonly service: ModelOrchestrationService) {}

  public queueJob = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const parsed = queueJobSchema.safeParse(req.body);
      if (!parsed.success) {
        throw new ValidationError("Invalid queue job payload", parsed.error.format());
      }
      const job = await this.service.queueJob(parsed.data);
      res.status(202).json({ success: true, data: job });
    } catch (err) {
      next(err);
    }
  };

  public getJob = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { jobId } = req.params;
      const job = await this.service.getJobStatus(jobId);
      res.status(200).json({ success: true, data: job });
    } catch (err) {
      next(err);
    }
  };

  public getMatchJobs = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { matchId } = req.params;
      const jobs = await this.service.getJobsForMatch(matchId);
      res.status(200).json({ success: true, data: jobs });
    } catch (err) {
      next(err);
    }
  };
}
