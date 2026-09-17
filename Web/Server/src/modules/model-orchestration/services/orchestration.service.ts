import { IJobRepository, AIJobRecord } from "../repositories/job.repository";
import { eventBus } from "../../../shared/events/event-bus";
import { getTracer } from "../../../observability/tracer";
import { logger } from "../../../observability/logger";
import { NotFoundError } from "../../../shared/errors/app-error";

export interface QueueJobDTO {
  matchId: string;
  videoId: string;
  pipelineType?: AIJobRecord["pipelineType"];
}

export class ModelOrchestrationService {
  constructor(private readonly jobRepo: IJobRepository) {
    this.initEventListeners();
  }

  /**
   * Decoupled integration: Automatically queue analysis when a video upload is confirmed.
   */
  private initEventListeners(): void {
    eventBus.subscribe<{ videoId: string; matchId: string; storagePath: string }>(
      "VIDEO_UPLOADED",
      async (event) => {
        logger.info(`[ModelOrchestration] Received VIDEO_UPLOADED event for video ${event.payload.videoId}`);
        await this.queueJob({
          matchId: event.payload.matchId,
          videoId: event.payload.videoId,
          pipelineType: "full_match_analysis",
        });
      }
    );
  }

  /**
   * Queues an AI inference job.
   */
  public async queueJob(dto: QueueJobDTO): Promise<AIJobRecord> {
    const tracer = getTracer("model-orchestration");
    return tracer.startActiveSpan("ModelOrchestrationService.queueJob", async (span) => {
      try {
        const jobId = `job_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
        span.setAttribute("job.id", jobId);
        span.setAttribute("job.matchId", dto.matchId);

        const job: AIJobRecord = {
          jobId,
          matchId: dto.matchId,
          videoId: dto.videoId,
          pipelineType: dto.pipelineType || "full_match_analysis",
          status: "queued",
          progressPercent: 0,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        };

        await this.jobRepo.saveJob(job);
        logger.info(`Queued AI pipeline job ${jobId}`, { matchId: dto.matchId });

        // Simulate asynchronous background worker progress
        this.processJobAsync(jobId, dto.matchId);

        return job;
      } finally {
        span.end();
      }
    });
  }

  /**
   * Simulates/polls pipeline execution, advancing progress and notifying domain listeners upon completion.
   */
  private processJobAsync(jobId: string, matchId: string): void {
    setTimeout(async () => {
      await this.jobRepo.updateJob(jobId, { status: "in_progress", progressPercent: 45 });

      setTimeout(async () => {
        await this.jobRepo.updateJob(jobId, {
          status: "completed",
          progressPercent: 100,
          resultPayload: {
            detectedPlayersCount: 22,
            ballTrackPoints: 1840,
            tacticalMatricesGenerated: true,
          },
        });

        logger.info(`AI pipeline job ${jobId} completed successfully`);

        // Emit domain event for match-analytics and stats-aggregation modules
        await eventBus.publish("MATCH_ANALYSIS_COMPLETED", {
          jobId,
          matchId,
          completedAt: new Date().toISOString(),
        });
      }, 3000);
    }, 1500);
  }

  public async getJobStatus(jobId: string): Promise<AIJobRecord> {
    const job = await this.jobRepo.getJob(jobId);
    if (!job) {
      throw new NotFoundError("Job", jobId);
    }
    return job;
  }

  public async getJobsForMatch(matchId: string): Promise<AIJobRecord[]> {
    return this.jobRepo.listJobsByMatch(matchId);
  }
}
