/**
 * Standard Result envelope for API and Service layer responses.
 */
export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

export function ok<T>(data: T): Result<T, never> {
  return { success: true, data };
}

export function fail<E = Error>(error: E): Result<never, E> {
  return { success: false, error };
}

/**
 * Standard pagination query parameters and result container.
 */
export interface PaginationParams {
  page?: number;
  limit?: number;
}

export interface PaginatedResult<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  totalPages: number;
}

/**
 * Common pitch coordinates [0, 105] length, [0, 68] width (meters standard)
 */
export interface PitchCoordinate {
  x: number;
  y: number;
  vx?: number;
  vy?: number;
}

/**
 * Base domain event contract.
 */
export interface DomainEvent<T = unknown> {
  eventId: string;
  eventName: string;
  occurredAt: Date;
  payload: T;
}
