import React from "react";
import { MatchEventItem } from "../../types/match";
import { Badge } from "../../components/Badge";
import { ArrowRightCircle, Target, Shield, AlertTriangle } from "lucide-react";

interface LiveEventFeedProps {
  events: MatchEventItem[];
}

export const LiveEventFeed: React.FC<LiveEventFeedProps> = ({ events }) => {
  const getEventIcon = (type: MatchEventItem["type"]) => {
    switch (type) {
      case "pass":
        return <ArrowRightCircle className="w-3.5 h-3.5 text-[#00B2FE]" />;
      case "shot":
        return <Target className="w-3.5 h-3.5 text-[#00F59B]" />;
      case "tackle":
      case "interception":
        return <Shield className="w-3.5 h-3.5 text-[#FFB020]" />;
      default:
        return <AlertTriangle className="w-3.5 h-3.5 text-[#FF4D4D]" />;
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between pb-3 border-b border-[#233354]">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Live Event Detection Feed
        </h4>
        <span className="text-[11px] font-mono text-slate-500">AI Verified</span>
      </div>

      <div className="flex-1 overflow-y-auto space-y-2.5 pt-3 pr-1">
        {events.map((evt) => (
          <div
            key={evt.id}
            className="p-2.5 rounded-lg bg-[#182238]/70 border border-[#233354]/80 hover:border-[#233354] transition-colors flex items-start gap-2.5"
          >
            <div className="mt-0.5">{getEventIcon(evt.type)}</div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-1">
                <span className="text-xs font-semibold text-slate-200 truncate">
                  {evt.player}
                </span>
                <span className="text-[10px] font-mono text-slate-400">
                  {String(evt.minute).padStart(2, "0")}:{String(evt.second).padStart(2, "0")}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-0.5">{evt.detail}</p>
              <div className="mt-1.5 flex items-center gap-1.5">
                <Badge variant={evt.team === "home" ? "info" : "danger"}>
                  {evt.team === "home" ? "Home" : "Away"}
                </Badge>
                <Badge variant={evt.success ? "success" : "warning"}>
                  {evt.success ? "Success" : "Incomplete"}
                </Badge>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
