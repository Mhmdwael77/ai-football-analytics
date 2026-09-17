import React from "react";
import { Header } from "./components/Header";
import { StatCards } from "./features/stat-widgets/StatCards";
import { TacticalPitch } from "./features/tactical-board/TacticalPitch";
import { LiveEventFeed } from "./features/live-feed/LiveEventFeed";
import { Card } from "./components/Card";
import { Button } from "./components/Button";
import { Badge } from "./components/Badge";
import { useMatchStore } from "./store/useMatchStore";
import { useMatchTelemetry } from "./hooks/useMatchTelemetry";
import { Play, Pause, RotateCcw, Layers, Eye, Compass, UserCheck } from "lucide-react";

export const App: React.FC = () => {
  const {
    currentFrame,
    isPlaying,
    togglePlay,
    setCurrentFrame,
    showPitchControl,
    togglePitchControl,
    showVelocityVectors,
    toggleVelocityVectors,
    selectedPlayerId,
  } = useMatchStore();

  const { frame, stats, events } = useMatchTelemetry();

  const selectedPlayer = frame.players.find((p) => p.playerId === selectedPlayerId);

  return (
    <div className="min-h-screen bg-[#0B0F19] text-slate-100 flex flex-col font-sans">
      <Header />

      <main className="flex-1 p-4 md:p-6 max-w-[1700px] w-full mx-auto flex flex-col gap-4">
        {/* Match Header & Score Banner */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 bg-[#121A2B] border border-[#233354] rounded-xl px-5 py-3">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg text-white">Manchester City</span>
              <Badge variant="info">HOME</Badge>
            </div>
            <div className="px-3 py-1 rounded bg-[#0B0F19] border border-[#233354] font-mono font-bold text-lg text-[#00F59B]">
              2 - 1
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="danger">AWAY</Badge>
              <span className="font-bold text-lg text-white">Arsenal</span>
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
            <span>Match Time: <b>64:18</b></span>
            <span>•</span>
            <span>Frame: <b>#{currentFrame}</b></span>
            <span>•</span>
            <Badge variant="success">Pitch Calibrated</Badge>
          </div>
        </div>

        {/* Top Stat Widgets */}
        <StatCards stats={stats} />

        {/* Center Dashboard: Main Pitch Canvas (Left/Center) + Sidebar (Right) */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 flex-1 items-start">
          {/* Main Pitch Canvas Area (Span 3 Columns) */}
          <div className="lg:col-span-3 flex flex-col gap-3">
            <div className="relative">
              <TacticalPitch frame={frame} />

              {/* Pitch Canvas Overlay Toolbar */}
              <div className="absolute top-4 left-4 flex flex-wrap items-center gap-2 bg-[#0B0F19]/85 backdrop-blur-md p-1.5 rounded-lg border border-[#233354]">
                <Button
                  size="sm"
                  variant={showPitchControl ? "primary" : "secondary"}
                  onClick={togglePitchControl}
                  className="gap-1.5"
                >
                  <Layers className="w-3.5 h-3.5" />
                  Pitch Control
                </Button>

                <Button
                  size="sm"
                  variant={showVelocityVectors ? "primary" : "secondary"}
                  onClick={toggleVelocityVectors}
                  className="gap-1.5"
                >
                  <Compass className="w-3.5 h-3.5" />
                  Velocity Vectors
                </Button>
              </div>
            </div>

            {/* Playback Controls & Frame Timeline Scrubber */}
            <Card className="bg-[#121A2B]/90">
              <div className="flex items-center gap-4">
                <Button
                  size="sm"
                  variant="primary"
                  onClick={togglePlay}
                  className="w-10 h-10 p-0 rounded-full"
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
                </Button>

                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => setCurrentFrame(0)}
                  className="w-8 h-8 p-0 rounded-full"
                  title="Restart playback"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </Button>

                <div className="flex-1 flex items-center gap-3">
                  <span className="text-xs font-mono text-slate-400">00:00</span>
                  <input
                    type="range"
                    min="0"
                    max="500"
                    value={currentFrame}
                    onChange={(e) => setCurrentFrame(parseInt(e.target.value, 10))}
                    className="w-full accent-[#00F59B] h-1.5 bg-[#233354] rounded-lg cursor-pointer"
                  />
                  <span className="text-xs font-mono text-slate-400">90:00</span>
                </div>

                <div className="text-xs font-mono px-3 py-1 rounded bg-[#0B0F19] border border-[#233354] text-[#00F59B]">
                  F:{currentFrame}
                </div>
              </div>
            </Card>
          </div>

          {/* Right Sidebar: Selected Player Inspector & Live Detections Feed */}
          <div className="flex flex-col gap-4 h-full">
            {/* Selected Player Tactical Card */}
            <Card title="Player Telemetry" className="bg-[#121A2B]/90">
              {selectedPlayer ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-8 rounded-full bg-[#00B2FE] flex items-center justify-center font-bold text-white text-xs">
                        #{selectedPlayer.jerseyNumber}
                      </div>
                      <div>
                        <div className="text-xs font-bold text-white">Player #{selectedPlayer.jerseyNumber}</div>
                        <div className="text-[11px] text-slate-400 capitalize">{selectedPlayer.team} Team</div>
                      </div>
                    </div>
                    <Badge variant={selectedPlayer.team === "home" ? "info" : "danger"}>
                      <UserCheck className="w-3 h-3 mr-1" /> Active
                    </Badge>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs bg-[#0B0F19] p-2.5 rounded-lg border border-[#233354]">
                    <div>
                      <span className="text-slate-500 block text-[10px]">Speed</span>
                      <span className="font-mono font-bold text-slate-200">
                        {(selectedPlayer.speedMps * 3.6).toFixed(1)} km/h
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-500 block text-[10px]">Pitch Pos</span>
                      <span className="font-mono font-bold text-slate-200">
                        {selectedPlayer.position.x.toFixed(1)}m, {selectedPlayer.position.y.toFixed(1)}m
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-5 text-xs text-slate-400">
                  <Eye className="w-6 h-6 mx-auto mb-2 text-slate-600" />
                  Click any player node on the pitch canvas to inspect live telemetry.
                </div>
              )}
            </Card>

            {/* Live Detection Feed */}
            <Card className="flex-1 bg-[#121A2B]/90 min-h-[380px]">
              <LiveEventFeed events={events} />
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
};
