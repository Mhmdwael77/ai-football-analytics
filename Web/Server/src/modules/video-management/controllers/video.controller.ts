import { Request, Response, NextFunction } from "express";
import { z } from "zod";
import { VideoService } from "../services/video.service";
import { ValidationError } from "../../../shared/errors/app-error";

const uploadUrlSchema = z.object({
  matchId: z.string().min(1, "matchId is required"),
  filename: z.string().min(1, "filename is required"),
  contentType: z.string().default("video/mp4"),
  sizeBytes: z.number().positive(),
});

export class VideoController {
  constructor(private readonly videoService: VideoService) {}

  public getUploadUrl = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const parsed = uploadUrlSchema.safeParse(req.body);
      if (!parsed.success) {
        throw new ValidationError("Invalid upload request payload", parsed.error.format());
      }
      const result = await this.videoService.generateUploadUrl(parsed.data);
      res.status(200).json({ success: true, data: result });
    } catch (err) {
      next(err);
    }
  };

  public confirmUpload = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { videoId } = req.params;
      const result = await this.videoService.confirmUpload(videoId);
      res.status(200).json({ success: true, data: result });
    } catch (err) {
      next(err);
    }
  };

  public getVideo = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { videoId } = req.params;
      const result = await this.videoService.getVideo(videoId);
      res.status(200).json({ success: true, data: result });
    } catch (err) {
      next(err);
    }
  };
}
