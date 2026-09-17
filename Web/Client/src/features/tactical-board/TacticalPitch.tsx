import React, { useRef, useEffect } from "react";
import * as d3 from "d3";
import { useMatchStore } from "../../store/useMatchStore";
import { TacticalFrameSnapshot } from "../../types/match";

interface TacticalPitchProps {
  frame: TacticalFrameSnapshot;
}

export const TacticalPitch: React.FC<TacticalPitchProps> = ({ frame }) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const {
    selectedPlayerId,
    setSelectedPlayerId,
    showPitchControl,
    showVelocityVectors,
  } = useMatchStore();

  // Pitch standard dimensions in meters
  const PITCH_WIDTH = 105;
  const PITCH_HEIGHT = 68;

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);

    // Clear dynamic layers on each frame update (leaving pitch base)
    svg.select("#dynamic-layer").selectAll("*").remove();
    const dynamicGroup = svg.select("#dynamic-layer");

    // 1. Optional Pitch Control Overlay (Grid Simulation)
    if (showPitchControl) {
      const cols = 21;
      const rows = 14;
      const cellW = PITCH_WIDTH / cols;
      const cellH = PITCH_HEIGHT / rows;

      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          const x = i * cellW;
          const y = j * cellH;
          // Calculate simulated home control gradient
          const distToHomeCenter = Math.hypot(x - 35, y - 34);
          const distToAwayCenter = Math.hypot(x - 70, y - 34);
          const homeProb = Math.min(Math.max(distToAwayCenter / (distToHomeCenter + distToAwayCenter), 0), 1);

          dynamicGroup
            .append("rect")
            .attr("x", x)
            .attr("y", y)
            .attr("width", cellW)
            .attr("height", cellH)
            .attr("fill", homeProb > 0.5 ? "#00B2FE" : "#FF4D4D")
            .attr("opacity", Math.abs(homeProb - 0.5) * 0.28);
        }
      }
    }

    // 2. Render Velocity Vectors
    if (showVelocityVectors) {
      frame.players.forEach((player) => {
        const vx = player.team === "home" ? 2.5 : -2.2;
        const vy = Math.sin(player.position.x) * 1.5;

        dynamicGroup
          .append("line")
          .attr("x1", player.position.x)
          .attr("y1", player.position.y)
          .attr("x2", player.position.x + vx)
          .attr("y2", player.position.y + vy)
          .attr("stroke", player.team === "home" ? "#00F59B" : "#FFB020")
          .attr("stroke-width", 0.4)
          .attr("stroke-dasharray", "0.6,0.6")
          .attr("opacity", 0.8);
      });
    }

    // 3. Render Players
    frame.players.forEach((player) => {
      const isSelected = selectedPlayerId === player.playerId;
      const isHome = player.team === "home";
      const playerColor = isHome ? "#00B2FE" : "#FF4D4D";

      const g = dynamicGroup
        .append("g")
        .attr("transform", `translate(${player.position.x}, ${player.position.y})`)
        .style("cursor", "pointer")
        .on("click", () => {
          setSelectedPlayerId(isSelected ? null : player.playerId);
        });

      // Selection Halo
      if (isSelected) {
        g.append("circle")
          .attr("r", 2.6)
          .attr("fill", "none")
          .attr("stroke", "#00F59B")
          .attr("stroke-width", 0.6)
          .attr("stroke-dasharray", "1,1");
      }

      // Player circle
      g.append("circle")
        .attr("r", 1.6)
        .attr("fill", playerColor)
        .attr("stroke", "#ffffff")
        .attr("stroke-width", 0.4)
        .attr("filter", "drop-shadow(0 2px 4px rgba(0,0,0,0.5))");

      // Jersey Number
      g.append("text")
        .attr("text-anchor", "middle")
        .attr("dy", "0.45em")
        .attr("font-size", "1.1px")
        .attr("font-weight", "bold")
        .attr("fill", "#ffffff")
        .text(player.jerseyNumber);
    });

    // 4. Render Ball
    const ballGroup = dynamicGroup
      .append("g")
      .attr("transform", `translate(${frame.ball.x}, ${frame.ball.y})`);

    // Ball glow
    ballGroup
      .append("circle")
      .attr("r", 1.8)
      .attr("fill", "#00F59B")
      .attr("opacity", 0.35);

    // Ball core
    ballGroup
      .append("circle")
      .attr("r", 0.9)
      .attr("fill", "#FFFFFF")
      .attr("stroke", "#000000")
      .attr("stroke-width", 0.2);

  }, [frame, selectedPlayerId, showPitchControl, showVelocityVectors, setSelectedPlayerId]);

  return (
    <div className="relative w-full h-full flex items-center justify-center p-2 bg-[#0d1424] rounded-xl overflow-hidden border border-[#233354]">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${PITCH_WIDTH} ${PITCH_HEIGHT}`}
        className="w-full h-full max-h-[640px] drop-shadow-2xl select-none"
      >
        <defs>
          <linearGradient id="grassGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#143823" />
            <stop offset="100%" stopColor="#0e2a1a" />
          </linearGradient>
        </defs>

        {/* Pitch Turf Base */}
        <rect width={PITCH_WIDTH} height={PITCH_HEIGHT} fill="url(#grassGrad)" rx="3" />

        {/* Pitch Markings Group */}
        <g stroke="#ffffff" strokeWidth="0.3" fill="none" opacity="0.65">
          {/* Outer Touchlines */}
          <rect x="0" y="0" width={PITCH_WIDTH} height={PITCH_HEIGHT} rx="2" />

          {/* Halfway Line */}
          <line x1={PITCH_WIDTH / 2} y1="0" x2={PITCH_WIDTH / 2} y2={PITCH_HEIGHT} />

          {/* Center Circle & Spot */}
          <circle cx={PITCH_WIDTH / 2} cy={PITCH_HEIGHT / 2} r="9.15" />
          <circle cx={PITCH_WIDTH / 2} cy={PITCH_HEIGHT / 2} r="0.4" fill="#ffffff" />

          {/* Left Penalty Box */}
          <rect x="0" y={(PITCH_HEIGHT - 40.3) / 2} width="16.5" height="40.3" />
          <rect x="0" y={(PITCH_HEIGHT - 18.3) / 2} width="5.5" height="18.3" />
          <circle cx="11.0" cy={PITCH_HEIGHT / 2} r="0.4" fill="#ffffff" />
          <path d={`M 16.5 ${PITCH_HEIGHT / 2 - 7.3} A 9.15 9.15 0 0 1 16.5 ${PITCH_HEIGHT / 2 + 7.3}`} />

          {/* Right Penalty Box */}
          <rect x={PITCH_WIDTH - 16.5} y={(PITCH_HEIGHT - 40.3) / 2} width="16.5" height="40.3" />
          <rect x={PITCH_WIDTH - 5.5} y={(PITCH_HEIGHT - 18.3) / 2} width="5.5" height="18.3" />
          <circle cx={PITCH_WIDTH - 11.0} cy={PITCH_HEIGHT / 2} r="0.4" fill="#ffffff" />
          <path d={`M ${PITCH_WIDTH - 16.5} ${PITCH_HEIGHT / 2 - 7.3} A 9.15 9.15 0 0 0 ${PITCH_WIDTH - 16.5} ${PITCH_HEIGHT / 2 + 7.3}`} />

          {/* Corner Arcs */}
          <path d="M 0 1 A 1 1 0 0 0 1 0" />
          <path d={`M 0 ${PITCH_HEIGHT - 1} A 1 1 0 0 1 1 ${PITCH_HEIGHT}`} />
          <path d={`M ${PITCH_WIDTH} 1 A 1 1 0 0 1 ${PITCH_WIDTH - 1} 0`} />
          <path d={`M ${PITCH_WIDTH} ${PITCH_HEIGHT - 1} A 1 1 0 0 0 ${PITCH_WIDTH - 1} ${PITCH_HEIGHT}`} />
        </g>

        {/* Dynamic Telemetry & Overlays Layer */}
        <g id="dynamic-layer" />
      </svg>
    </div>
  );
};
