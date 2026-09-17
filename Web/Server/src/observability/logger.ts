import { trace, context } from "@opentelemetry/api";

export type LogLevel = "debug" | "info" | "warn" | "error";

interface LogPayload {
  level: LogLevel;
  message: string;
  timestamp: string;
  traceId?: string;
  spanId?: string;
  context?: Record<string, unknown>;
  error?: {
    message: string;
    stack?: string;
  };
}

class Logger {
  private formatLog(level: LogLevel, message: string, ctx?: Record<string, unknown>, err?: Error): LogPayload {
    const currentSpan = trace.getSpan(context.active());
    const spanContext = currentSpan?.spanContext();

    const payload: LogPayload = {
      level,
      message,
      timestamp: new Date().toISOString(),
      traceId: spanContext?.traceId,
      spanId: spanContext?.spanId,
      context: ctx,
    };

    if (err) {
      payload.error = {
        message: err.message,
        stack: err.stack,
      };
    }

    return payload;
  }

  public info(message: string, ctx?: Record<string, unknown>): void {
    const log = this.formatLog("info", message, ctx);
    console.log(JSON.stringify(log));
  }

  public warn(message: string, ctx?: Record<string, unknown>): void {
    const log = this.formatLog("warn", message, ctx);
    console.warn(JSON.stringify(log));
  }

  public error(message: string, err?: Error, ctx?: Record<string, unknown>): void {
    const log = this.formatLog("error", message, ctx, err);
    console.error(JSON.stringify(log));
  }

  public debug(message: string, ctx?: Record<string, unknown>): void {
    if (process.env.NODE_ENV !== "production") {
      const log = this.formatLog("debug", message, ctx);
      console.debug(JSON.stringify(log));
    }
  }
}

export const logger = new Logger();
