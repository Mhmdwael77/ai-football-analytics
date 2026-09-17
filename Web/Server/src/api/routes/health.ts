import { Router, Request, Response } from "express";
import { checkRedisHealth } from "../../config/redis";
import { checkSupabaseHealth } from "../../config/supabase";

export const healthRouter = Router();

healthRouter.get("/", async (_req: Request, res: Response) => {
  const redisStatus = await checkRedisHealth();
  const supabaseStatus = await checkSupabaseHealth();

  const isHealthy = redisStatus.status === "connected" || supabaseStatus.status === "connected";

  const payload = {
    status: isHealthy ? "ok" : "degraded",
    timestamp: new Date().toISOString(),
    uptimeSeconds: Math.floor(process.uptime()),
    memoryUsageMB: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
    services: {
      redis: redisStatus,
      supabase: supabaseStatus,
    },
  };

  res.status(isHealthy ? 200 : 503).json(payload);
});
