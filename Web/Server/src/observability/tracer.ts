import { NodeSDK } from "@opentelemetry/sdk-node";
import { getNodeAutoInstrumentations } from "@opentelemetry/auto-instrumentations-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { Resource } from "@opentelemetry/resources";
import { SemanticResourceAttributes } from "@opentelemetry/semantic-conventions";
import { trace, Tracer } from "@opentelemetry/api";
import { env } from "../config/env";

let sdk: NodeSDK | null = null;

/**
 * Bootstraps OpenTelemetry NodeSDK before importing Express or other modules.
 */
export function initOpenTelemetry(): void {
  if (sdk) {
    return;
  }

  const traceExporter = new OTLPTraceExporter({
    url: env.OTEL_EXPORTER_OTLP_ENDPOINT,
  });

  sdk = new NodeSDK({
    resource: new Resource({
      [SemanticResourceAttributes.SERVICE_NAME]: env.OTEL_SERVICE_NAME,
      [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: env.NODE_ENV,
      [SemanticResourceAttributes.SERVICE_VERSION]: "1.0.0",
    }),
    traceExporter,
    instrumentations: [
      getNodeAutoInstrumentations({
        "@opentelemetry/instrumentation-fs": {
          enabled: false, // Disables high-frequency file system spans
        },
      }),
    ],
  });

  try {
    sdk.start();
    console.log(`[OpenTelemetry] SDK initialized with endpoint ${env.OTEL_EXPORTER_OTLP_ENDPOINT}`);
  } catch (error) {
    console.error("[OpenTelemetry] Initialization error:", error);
  }

  process.on("SIGTERM", () => {
    sdk?.shutdown()
      .then(() => console.log("[OpenTelemetry] SDK shut down successfully"))
      .catch((err) => console.error("[OpenTelemetry] Error shutting down SDK", err))
      .finally(() => process.exit(0));
  });
}

/**
 * Returns an application-scoped Tracer instance.
 */
export function getTracer(name = "trakora-core"): Tracer {
  return trace.getTracer(name);
}
