import { Router } from "express";
import { VideoRepository, IVideoRepository, VideoRecord } from "./repositories/video.repository";
import { VideoService, CreateUploadUrlDTO, UploadUrlResult } from "./services/video.service";
import { VideoController } from "./controllers/video.controller";

export { IVideoRepository, VideoRepository, VideoRecord, VideoService, CreateUploadUrlDTO, UploadUrlResult };

export function createVideoManagementModule(): { router: Router; service: VideoService } {
  const repository = new VideoRepository();
  const service = new VideoService(repository);
  const controller = new VideoController(service);

  const router = Router();

  router.post("/upload-url", controller.getUploadUrl);
  router.post("/:videoId/confirm", controller.confirmUpload);
  router.get("/:videoId", controller.getVideo);

  return { router, service };
}
