import { getRedisClient } from "../../../config/redis";
import { logger } from "../../../observability/logger";

export interface AIJobRecord {
  jobId: string;
  matchId: string;
  videoId: string;
  pipelineType: "detection" | "tracking" | "pitch_calibration" | "full_match_analysis";
  status: "queued" | "in_progress" | "completed" | "failed";
  progressPercent: number;
  resultPayload?: Record<string, unknown>;
  error?: string;
  createdAt: string;
  updatedAt: string;
}

export interface IJobRepository {
  saveJob(job: AIJobRecord): Promise<void>;
  getJob(jobId: string): Promise<AIJobRecord | null>;
  updateJob(jobId: string, partial: Partial<AIJobRecord>): Promise<AIJobRecord | null>;
  listJobsByMatch(matchId: string): Promise<AIJobRecord[]>;
}

export class JobRepository implements IJobRepository {
  private inMemoryFallback = new Map<string, AIJobRecord>();

  private getKey(jobId: string): string {
    return `trakora:job:${jobId}`;
  }

  public async saveJob(job: AIJobRecord): Promise<void> {
    this.inMemoryFallback.set(job.jobId, job);
    try {
      const redis = getRedisClient();
      await redis.set(this.getKey(job.jobId), JSON.stringify(job), "EX", 86400); // 24h TTL
      await redis.sadd(`trakora:match_jobs:${job.matchId}`, job.jobId);
    } catch (err) {
      logger.warn("Redis saveJob failed, preserved in memory fallback", { error: (err as Error).message });
    }
  }

  public async getJob(jobId: string): Promise<AIJobRecord | null> {
    try {
      const redis = getRedisClient();
      const raw = await redis.get(this.getKey(jobId));
      if (raw) {
        return JSON.parse(raw) as AIJobRecord;
      }
    } catch (err) {
      logger.warn("Redis getJob failed, falling back to in-memory", { error: (err as Error).message });
    }
    return this.inMemoryFallback.get(jobId) || null;
  }

  public async updateJob(jobId: string, partial: Partial<AIJobRecord>): Promise<AIJobRecord | null> {
    const existing = await this.getJob(jobId);
    if (!existing) return null;

    const updated: AIJobRecord = {
      ...existing,
      ...partial,
      updatedAt: new Date().toISOString(),
    };

    await this.saveJob(updated);
    return updated;
  }

  public async listJobsByMatch(matchId: string): Promise<AIJobRecord[]> {
    try {
      const redis = getRedisClient();
      const jobIds = await redis.smembers(`trakora:match_jobs:${matchId}`);
      if (jobIds && jobIds.length > 0) {
        const jobs: AIJobRecord[] = [];
        for (const id of jobIds) {
          const j = await this.getJob(id);
          if (j) jobs.push(j);
        }
        return jobs;
      }
    } catch {
      // Fallback
    }
    return Array.from(this.inMemoryFallback.values()).filter((j) => j.matchId === matchId);
  }
}
