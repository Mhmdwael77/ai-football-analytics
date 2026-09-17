import React from "react";
import { MatchStatsSummary } from "../../types/match";
import { Card } from "../../components/Card";
import { Activity, Flame, TrendingUp, Compass } from "lucide-react";

interface StatCardsProps {
  stats: MatchStatsSummary;
}

export const StatCards: React.FC<StatCardsProps> = ({ stats }) => {
  const { homeTeam, awayTeam, turnovers, intensityIndex } = stats;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {/* Possession Metric Card */}
      <Card className="bg-[#121A2B]/90">
        <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
          <span className="flex items-center gap-1 font-medium">
            <Compass className="w-3.5 h-3.5 text-[#00B2FE]" /> Possession
          </span>
          <span className="font-mono text-slate-200">
            {homeTeam.possessionPercent}% - {awayTeam.possessionPercent}%
          </span>
        </div>
        <div className="w-full h-2 bg-[#182238] rounded-full overflow-hidden flex mt-2">
          <div
            className="bg-[#00B2FE] h-full transition-all duration-300"
            style={{ width: `${homeTeam.possessionPercent}%` }}
          />
          <div
            className="bg-[#FF4D4D] h-full transition-all duration-300"
            style={{ width: `${awayTeam.possessionPercent}%` }}
          />
        </div>
      </Card>

      {/* Expected Goals (xG) */}
      <Card className="bg-[#121A2B]/90">
        <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
          <span className="flex items-center gap-1 font-medium">
            <TrendingUp className="w-3.5 h-3.5 text-[#00F59B]" /> Expected Goals (xG)
          </span>
        </div>
        <div className="flex items-baseline justify-between mt-1">
          <span className="text-xl font-bold font-mono text-[#00B2FE]">
            {homeTeam.expectedGoals.toFixed(2)}
          </span>
          <span className="text-xs text-slate-500 font-mono">VS</span>
          <span className="text-xl font-bold font-mono text-[#FF4D4D]">
            {awayTeam.expectedGoals.toFixed(2)}
          </span>
        </div>
      </Card>

      {/* Pass Accuracy */}
      <Card className="bg-[#121A2B]/90">
        <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
          <span className="flex items-center gap-1 font-medium">
            <Activity className="w-3.5 h-3.5 text-[#FFB020]" /> Pass Accuracy
          </span>
          <span className="text-xs font-mono text-emerald-400">
            {homeTeam.passAccuracyPercent}%
          </span>
        </div>
        <div className="flex items-center justify-between text-xs text-slate-400 mt-2">
          <span>{homeTeam.successfulPasses}/{homeTeam.totalPasses} Completed</span>
          <span className="text-slate-500">{awayTeam.passAccuracyPercent}% Away</span>
        </div>
      </Card>

      {/* Match Intensity & Turnovers */}
      <Card className="bg-[#121A2B]/90">
        <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
          <span className="flex items-center gap-1 font-medium">
            <Flame className="w-3.5 h-3.5 text-amber-500" /> Pressing Intensity
          </span>
          <span className="text-xs font-bold text-amber-400">{intensityIndex}/100</span>
        </div>
        <div className="flex items-center justify-between text-xs text-slate-400 mt-2">
          <span>Turnovers forced: <b className="text-slate-200">{turnovers}</b></span>
          <span className="text-slate-500">Distance: {homeTeam.distanceCoveredKm}km</span>
        </div>
      </Card>
    </div>
  );
};
