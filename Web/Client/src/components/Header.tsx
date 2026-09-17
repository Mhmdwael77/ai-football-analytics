import React from "react";
import { Activity, ShieldCheck } from "lucide-react";
import { Badge } from "./Badge";

export const Header: React.FC = () => {
  return (
    <header className="h-14 border-b border-[#233354] bg-[#0B0F19]/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-50">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-[#00F59B] to-[#00B2FE] flex items-center justify-center font-bold text-slate-950 text-base shadow-md shadow-[#00F59B]/20">
          T
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-white tracking-wider text-base">TRAKORA</span>
            <span className="text-xs text-slate-500 font-mono">v1.0-modular</span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="hidden md:flex items-center gap-2 px-3 py-1 rounded-full bg-[#121A2B] border border-[#233354]">
          <span className="w-2 h-2 rounded-full bg-[#00F59B] animate-pulse" />
          <span className="text-xs text-slate-300 font-medium">Model Pipeline: Online</span>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="info">
            <ShieldCheck className="w-3 h-3 mr-1" /> Match #01 vs Arsenal
          </Badge>
          <Badge variant="success">
            <Activity className="w-3 h-3 mr-1" /> OTEL Tracing Active
          </Badge>
        </div>
      </div>
    </header>
  );
};
