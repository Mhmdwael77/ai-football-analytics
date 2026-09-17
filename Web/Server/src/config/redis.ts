import Redis from "ioredis";
import { env } from "./env";
import { logger } from "../observability/logger";

let redisClient: Redis | null = null;

/**
 * Initializes and returns a singleton Redis client instance.
 */
export function getRedisClient(): Redis {
  if (!redisClient) {
    redisClient = new Redis(env.REDIS_URL, {
      maxRetriesPerRequest: 1,
      connectTimeout: 2000,
      enableOfflineQueue: true,
      retryStrategy(times) {
        if (times > 5) {
          return null; // Stop retrying after 5 attempts to avoid infinite loops when offline
        }
        return Math.min(times * 200, 1000);
      },
      lazyConnect: true,
    });

    redisClient.on("connect", () => {
      logger.info("Connected to Redis instance", { url: env.REDIS_URL });
    });

    redisClient.on("error", (err) => {
      logger.warn("Redis connection notice", { error: err.message });
    });
  }
  return redisClient;
}

/**
 * Connects to Redis asynchronously at startup without blocking server boot.
 */
export async function connectRedis(): Promise<void> {
  const client = getRedisClient();
  try {
    await Promise.race([
      client.connect(),
      new Promise((_, reject) => setTimeout(() => reject(new Error("Redis connection timeout")), 1500)),
    ]);
    logger.info("Redis eagerly connected.");
  } catch (err) {
    logger.warn("Redis initial eager connection skipped, will operate in fallback mode", {
      error: (err as Error).message,
    });
  }
}

/**
 * Check connectivity and responsiveness of the Redis instance.
 */
export async function checkRedisHealth(): Promise<{ status: "connected" | "disconnected"; latencyMs: number }> {
  const client = getRedisClient();
  const start = Date.now();
  try {
    const pong = await client.ping();
    const latencyMs = Date.now() - start;
    if (pong === "PONG") {
      return { status: "connected", latencyMs };
    }
    return { status: "disconnected", latencyMs };
  } catch {
    return { status: "disconnected", latencyMs: Date.now() - start };
  }
}
