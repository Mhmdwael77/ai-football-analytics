import { Request, Response, NextFunction } from "express";
import { MatchAnalyticsService } from "../services/analytics.service";

export class AnalyticsController {
  constructor(private readonly service: MatchAnalyticsService) {}

  public getTacticalFrame = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { matchId } = req.params;
      const frameNumber = parseInt(req.query.frame as string, 10) || 0;
      const frame = await this.service.getTacticalFrame(matchId, frameNumber);
      res.status(200).json({ success: true, data: frame });
    } catch (err) {
      next(err);
    }
  };

  public getPitchControl = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { matchId } = req.params;
      const frameNumber = parseInt(req.query.frame as string, 10) || 0;
      const grid = await this.service.getPitchControl(matchId, frameNumber);
      res.status(200).json({ success: true, data: grid });
    } catch (err) {
      next(err);
    }
  };

  public getPlayerHeatmap = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { matchId, playerId } = req.params;
      const heatmap = await this.service.getPlayerHeatmap(matchId, playerId);
      res.status(200).json({ success: true, data: heatmap });
    } catch (err) {
      next(err);
    }
  };
}
