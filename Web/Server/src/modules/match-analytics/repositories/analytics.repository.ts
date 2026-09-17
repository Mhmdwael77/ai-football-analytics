import { PitchCoordinate } from "../../../shared/types/common";

export interface PlayerTrackPoint {
  playerId: string;
  jerseyNumber: number;
  team: "home" | "away";
  position: PitchCoordinate;
  speedMps: number;
}

export interface TacticalFrameSnapshot {
  frameNumber: number;
  timestampSeconds: number;
  players: PlayerTrackPoint[];
  ball: PitchCoordinate & { z?: number };
}

export interface PitchControlGrid {
  gridX: number;
  gridY: number;
  homeControlValues: number[][]; // 0.0 to 1.0 (probability of home team control)
}

export interface IAnalyticsRepository {
  getFrame(matchId: string, frameNumber: number): Promise<TacticalFrameSnapshot | null>;
  saveFrame(matchId: string, frame: TacticalFrameSnapshot): Promise<void>;
  getPitchControl(matchId: string, frameNumber: number): Promise<PitchControlGrid>;
}

export class AnalyticsRepository implements IAnalyticsRepository {
  private frameStorage = new Map<string, TacticalFrameSnapshot[]>();

  public async getFrame(matchId: string, frameNumber: number): Promise<TacticalFrameSnapshot | null> {
    const frames = this.frameStorage.get(matchId);
    if (!frames || frames.length === 0) {
      return this.generateMockFrame(frameNumber);
    }
    const found = frames.find((f) => f.frameNumber === frameNumber);
    return found || this.generateMockFrame(frameNumber);
  }

  public async saveFrame(matchId: string, frame: TacticalFrameSnapshot): Promise<void> {
    const existing = this.frameStorage.get(matchId) || [];
    existing.push(frame);
    this.frameStorage.set(matchId, existing);
  }

  public async getPitchControl(_matchId: string, _frameNumber: number): Promise<PitchControlGrid> {
    const gridX = 32;
    const gridY = 24;
    const homeControlValues: number[][] = [];

    for (let y = 0; y < gridY; y++) {
      const row: number[] = [];
      for (let x = 0; x < gridX; x++) {
        // Gradient simulating attacking team presence on right side (x > 16)
        const baseProb = x / gridX;
        const variation = Math.sin((x + y) * 0.4) * 0.15;
        row.push(Math.min(Math.max(Number((baseProb + variation).toFixed(2)), 0.05), 0.95));
      }
      homeControlValues.push(row);
    }

    return { gridX, gridY, homeControlValues };
  }

  private generateMockFrame(frameNumber: number): TacticalFrameSnapshot {
    const players: PlayerTrackPoint[] = [];

    // 11 Home team players (4-3-3 formation simulation on 105x68 pitch)
    const homePositions = [
      { x: 8, y: 34, no: 1 },
      { x: 25, y: 12, no: 2 }, { x: 22, y: 26, no: 4 }, { x: 22, y: 42, no: 5 }, { x: 25, y: 56, no: 3 },
      { x: 42, y: 22, no: 8 }, { x: 38, y: 34, no: 6 }, { x: 44, y: 46, no: 10 },
      { x: 62, y: 15, no: 7 }, { x: 68, y: 34, no: 9 }, { x: 62, y: 53, no: 11 },
    ];

    homePositions.forEach((pos, idx) => {
      players.push({
        playerId: `h_${pos.no}`,
        jerseyNumber: pos.no,
        team: "home",
        position: {
          x: pos.x + Math.sin(frameNumber * 0.1 + idx) * 1.5,
          y: pos.y + Math.cos(frameNumber * 0.1 + idx) * 1.5,
        },
        speedMps: 3.2,
      });
    });

    // 11 Away team players (4-4-2 formation simulation)
    const awayPositions = [
      { x: 97, y: 34, no: 1 },
      { x: 80, y: 14, no: 2 }, { x: 83, y: 27, no: 4 }, { x: 83, y: 41, no: 5 }, { x: 80, y: 54, no: 3 },
      { x: 65, y: 16, no: 7 }, { x: 60, y: 28, no: 8 }, { x: 60, y: 40, no: 6 }, { x: 65, y: 52, no: 11 },
      { x: 48, y: 28, no: 9 }, { x: 48, y: 40, no: 10 },
    ];

    awayPositions.forEach((pos, idx) => {
      players.push({
        playerId: `a_${pos.no}`,
        jerseyNumber: pos.no,
        team: "away",
        position: {
          x: pos.x + Math.cos(frameNumber * 0.1 + idx) * 1.2,
          y: pos.y + Math.sin(frameNumber * 0.1 + idx) * 1.2,
        },
        speedMps: 2.8,
      });
    });

    return {
      frameNumber,
      timestampSeconds: frameNumber / 25,
      players,
      ball: {
        x: 52 + Math.sin(frameNumber * 0.05) * 10,
        y: 34 + Math.cos(frameNumber * 0.05) * 8,
        z: 0.2,
      },
    };
  }
}
