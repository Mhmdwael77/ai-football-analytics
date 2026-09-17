import { EventEmitter } from "events";
import { DomainEvent } from "../types/common";
import { logger } from "../../observability/logger";
import { getTracer } from "../../observability/tracer";

type EventHandler<T> = (event: DomainEvent<T>) => Promise<void> | void;

class EventBus {
  private emitter = new EventEmitter();

  constructor() {
    this.emitter.setMaxListeners(50);
  }

  /**
   * Publishes a domain event to all registered module listeners.
   */
  public async publish<T>(eventName: string, payload: T): Promise<void> {
    const tracer = getTracer("event-bus");
    return tracer.startActiveSpan(`event-bus.publish:${eventName}`, async (span) => {
      try {
        const event: DomainEvent<T> = {
          eventId: `evt_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
          eventName,
          occurredAt: new Date(),
          payload,
        };

        span.setAttribute("event.name", eventName);
        span.setAttribute("event.id", event.eventId);
        logger.info(`[EventBus] Publishing event ${eventName}`, { eventId: event.eventId });

        this.emitter.emit(eventName, event);
      } catch (err) {
        span.recordException(err as Error);
        logger.error(`[EventBus] Error publishing event ${eventName}`, err as Error);
      } finally {
        span.end();
      }
    });
  }

  /**
   * Subscribes a handler to a domain event.
   */
  public subscribe<T>(eventName: string, handler: EventHandler<T>): void {
    this.emitter.on(eventName, async (event: DomainEvent<T>) => {
      const tracer = getTracer("event-bus-subscriber");
      await tracer.startActiveSpan(`event-bus.handle:${eventName}`, async (span) => {
        try {
          span.setAttribute("event.id", event.eventId);
          await handler(event);
        } catch (err) {
          span.recordException(err as Error);
          logger.error(`[EventBus] Error handling event ${eventName}`, err as Error, {
            eventId: event.eventId,
          });
        } finally {
          span.end();
        }
      });
    });
    logger.info(`[EventBus] Subscribed to ${eventName}`);
  }
}

export const eventBus = new EventBus();
