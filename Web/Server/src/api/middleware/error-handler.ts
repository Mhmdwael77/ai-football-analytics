import { Request, Response, NextFunction } from "express";
import { AppError } from "../../shared/errors/app-error";
import { logger } from "../../observability/logger";
import { trace, context } from "@opentelemetry/api";

export function errorHandler(
  err: Error,
  _req: Request,
  res: Response,
  _next: NextFunction
): void {
  const currentSpan = trace.getSpan(context.active());
  currentSpan?.recordException(err);

  if (err instanceof AppError) {
    logger.warn(`Operational AppError: ${err.message}`, {
      statusCode: err.statusCode,
      details: err.details,
    });

    res.status(err.statusCode).json({
      success: false,
      error: {
        message: err.message,
        details: err.details,
      },
    });
    return;
  }

  logger.error(`Unhandled Internal Error: ${err.message}`, err);

  res.status(500).json({
    success: false,
    error: {
      message: process.env.NODE_ENV === "production" ? "Internal server error" : err.message,
    },
  });
}
