/**
 * CRITICAL: Initialize OpenTelemetry SDK FIRST before importing express or network libraries
 */
import { initOpenTelemetry } from "./observability/tracer";
initOpenTelemetry();

import express from "express";
import cors from "cors";
import { env } from "./config/env";
import { connectRedis } from "./config/redis";
import { logger } from "./observability/logger";
import { requestLogger } from "./api/middleware/request-logger";
import { errorHandler } from "./api/middleware/error-handler";
import { createApiRouter } from "./api/routes";

async function bootstrap(): Promise<void> {
  // Connect to Redis broker eagerly
  await connectRedis();

  const app = express();

  // Cross-Origin Resource Sharing
  app.use(
    cors({
      origin: env.CORS_ORIGIN === "*" ? true : env.CORS_ORIGIN.split(","),
      credentials: true,
    })
  );

  // Standard body parsers
  app.use(express.json({ limit: "25mb" }));
  app.use(express.urlencoded({ extended: true, limit: "25mb" }));

  // Observability request logging
  app.use(requestLogger);

  // Mount API gateway router
  app.use("/api", createApiRouter());

  // Global Error Handler
  app.use(errorHandler);

  // Start HTTP Server
  const server = app.listen(env.PORT, env.HOST, () => {
    logger.info(`Trakora Modular Monolith Server running on http://${env.HOST}:${env.PORT}`);
    logger.info(`Environment: ${env.NODE_ENV}`);
  });

  // Graceful Shutdown Handler
  const shutdown = async (signal: string) => {
    logger.info(`Received ${signal}. Shutting down gracefully...`);
    server.close(() => {
      logger.info("HTTP server closed.");
      process.exit(0);
    });

    // Force close after 10 seconds if hanging
    setTimeout(() => {
      logger.error("Could not close connections in time, forcefully shutting down");
      process.exit(1);
    }, 10000);
  };

  process.on("SIGTERM", () => shutdown("SIGTERM"));
  process.on("SIGINT", () => shutdown("SIGINT"));
}

bootstrap().catch((err) => {
  console.error("Fatal bootstrap failure:", err);
  process.exit(1);
});
