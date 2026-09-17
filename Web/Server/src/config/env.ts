import dotenv from "dotenv";
import { z } from "zod";

// Load environment variables from .env file
dotenv.config();

const envSchema = z.object({
  NODE_ENV: z.enum(["development", "test", "production"]).default("development"),
  PORT: z.coerce.number().default(4000),
  HOST: z.string().default("0.0.0.0"),
  CORS_ORIGIN: z.string().default("http://localhost:5173"),

  // Supabase Configuration
  SUPABASE_URL: z.string().url().default("https://placeholder-project.supabase.co"),
  SUPABASE_ANON_KEY: z.string().min(1).default("placeholder-anon-key"),
  SUPABASE_SERVICE_ROLE_KEY: z.string().optional(),
  SUPABASE_STORAGE_BUCKET: z.string().default("match-videos"),

  // Redis Configuration
  REDIS_URL: z.string().default("redis://localhost:6379"),
  UPSTASH_REDIS_REST_URL: z.string().optional(),
  UPSTASH_REDIS_REST_TOKEN: z.string().optional(),

  // OpenTelemetry Configuration
  OTEL_SERVICE_NAME: z.string().default("trakora-backend"),
  OTEL_EXPORTER_OTLP_ENDPOINT: z.string().default("http://localhost:4318/v1/traces"),
  OTEL_METRICS_EXPORTER_ENDPOINT: z.string().default("http://localhost:4318/v1/metrics"),
  OTEL_LOG_LEVEL: z.string().default("info"),
});

const parsed = envSchema.safeParse(process.env);

if (!parsed.success) {
  console.error("Invalid environment configuration:", parsed.error.format());
  process.exit(1);
}

export const env = parsed.data;
export type Env = z.infer<typeof envSchema>;
