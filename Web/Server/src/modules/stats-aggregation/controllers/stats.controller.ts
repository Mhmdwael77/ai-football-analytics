import { Request, Response, NextFunction } from "express";
import { StatsAggregationService } from "../services/stats.service";

export class StatsController {
  constructor(private readonly service: StatsAggregationService) {}

  public getMatchSummary = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { matchId } = req.params;
      const summary = await this.service.getMatchSummary(matchId);
      res.status(200).json({ success: true, data: summary });
    } catch (err) {
      next(err);
    }
  };

  public getPlayerStats = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { matchId } = req.params;
      const players = await this.service.getPlayerStats(matchId);
      res.status(200).json({ success: true, data: players });
    } catch (err) {
      next(err);
    }
  };
}
