import { createClient, SupabaseClient } from "@supabase/supabase-js";
import { env } from "./env";
import { logger } from "../observability/logger";

let supabaseClient: SupabaseClient | null = null;

/**
 * Initializes and returns the Supabase client instance.
 */
export function getSupabase(): SupabaseClient {
  if (!supabaseClient) {
    supabaseClient = createClient(env.SUPABASE_URL, env.SUPABASE_ANON_KEY, {
      auth: {
        persistSession: false,
      },
    });
    logger.info("Supabase client initialized", { endpoint: env.SUPABASE_URL });
  }
  return supabaseClient;
}

/**
 * Check connectivity and configuration status for Supabase.
 */
export async function checkSupabaseHealth(): Promise<{ status: "connected" | "misconfigured"; message: string }> {
  try {
    const isPlaceholder = env.SUPABASE_URL.includes("placeholder");
    if (isPlaceholder) {
      return {
        status: "misconfigured",
        message: "Supabase URL is configured with placeholder values.",
      };
    }
    const client = getSupabase();
    // Test listing buckets or basic ping
    const { error } = await client.storage.listBuckets();
    if (error) {
      return {
        status: "misconfigured",
        message: error.message,
      };
    }
    return {
      status: "connected",
      message: "Supabase client connected successfully.",
    };
  } catch (error) {
    return {
      status: "misconfigured",
      message: (error as Error).message,
    };
  }
}
