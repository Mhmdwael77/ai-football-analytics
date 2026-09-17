import { getSupabase } from "../../../config/supabase";
import { env } from "../../../config/env";
import { eventBus } from "../../../shared/events/event-bus";
import { getTracer } from "../../../observability/tracer";
import { NotFoundError } from "../../../shared/errors/app-error";
import { IVideoRepository, VideoRecord } from "../repositories/video.repository";

export interface CreateUploadUrlDTO {
  matchId: string;
  filename: string;
  contentType: string;
  sizeBytes: number;
}

export interface UploadUrlResult {
  videoId: string;
  uploadUrl: string;
  storagePath: string;
}

export class VideoService {
  constructor(private readonly videoRepo: IVideoRepository) {}

  /**
   * Generates a Supabase Storage signed upload URL and registers pending record.
   */
  public async generateUploadUrl(dto: CreateUploadUrlDTO): Promise<UploadUrlResult> {
    const tracer = getTracer("video-management");
    return tracer.startActiveSpan("VideoService.generateUploadUrl", async (span) => {
      try {
        const videoId = `vid_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
        const storagePath = `matches/${dto.matchId}/${videoId}_${dto.filename}`;

        span.setAttribute("video.id", videoId);
        span.setAttribute("video.matchId", dto.matchId);

        let uploadUrl = `${env.SUPABASE_URL}/storage/v1/object/${env.SUPABASE_STORAGE_BUCKET}/${storagePath}`;

        try {
          const supabase = getSupabase();
          const { data, error } = await supabase.storage
            .from(env.SUPABASE_STORAGE_BUCKET)
            .createSignedUploadUrl(storagePath);

          if (!error && data?.signedUrl) {
            uploadUrl = data.signedUrl;
          }
        } catch {
          // Fallback to direct path in mock/dev mode
        }

        await this.videoRepo.create({
          id: videoId,
          matchId: dto.matchId,
          filename: dto.filename,
          storagePath,
          contentType: dto.contentType,
          sizeBytes: dto.sizeBytes,
          status: "pending_upload",
        });

        return {
          videoId,
          uploadUrl,
          storagePath,
        };
      } finally {
        span.end();
      }
    });
  }

  /**
   * Confirms video upload has completed and triggers downstream AI pipeline via EventBus.
   */
  public async confirmUpload(videoId: string): Promise<VideoRecord> {
    const record = await this.videoRepo.findById(videoId);
    if (!record) {
      throw new NotFoundError("Video", videoId);
    }

    const updated = await this.videoRepo.updateStatus(videoId, "uploaded");

    // Decoupled communication: Publish domain event across monolith modules
    await eventBus.publish("VIDEO_UPLOADED", {
      videoId: record.id,
      matchId: record.matchId,
      storagePath: record.storagePath,
      uploadedAt: new Date().toISOString(),
    });

    return updated!;
  }

  public async getVideo(videoId: string): Promise<VideoRecord> {
    const record = await this.videoRepo.findById(videoId);
    if (!record) {
      throw new NotFoundError("Video", videoId);
    }
    return record;
  }
}
