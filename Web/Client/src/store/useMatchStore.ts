import { create } from "zustand";

interface MatchState {
  currentMatchId: string;
  currentFrame: number;
  isPlaying: boolean;
  selectedPlayerId: string | null;
  showPitchControl: boolean;
  showVoronoi: boolean;
  showVelocityVectors: boolean;

  setCurrentMatchId: (id: string) => void;
  setCurrentFrame: (frame: number) => void;
  incrementFrame: () => void;
  togglePlay: () => void;
  setSelectedPlayerId: (id: string | null) => void;
  togglePitchControl: () => void;
  toggleVoronoi: () => void;
  toggleVelocityVectors: () => void;
}

export const useMatchStore = create<MatchState>((set) => ({
  currentMatchId: "match_demo_01",
  currentFrame: 0,
  isPlaying: false,
  selectedPlayerId: null,
  showPitchControl: true,
  showVoronoi: false,
  showVelocityVectors: true,

  setCurrentMatchId: (id) => set({ currentMatchId: id }),
  setCurrentFrame: (frame) => set({ currentFrame: frame }),
  incrementFrame: () => set((state) => ({ currentFrame: (state.currentFrame + 1) % 500 })),
  togglePlay: () => set((state) => ({ isPlaying: !state.isPlaying })),
  setSelectedPlayerId: (id) => set({ selectedPlayerId: id }),
  togglePitchControl: () => set((state) => ({ showPitchControl: !state.showPitchControl })),
  toggleVoronoi: () => set((state) => ({ showVoronoi: !state.showVoronoi })),
  toggleVelocityVectors: () => set((state) => ({ showVelocityVectors: !state.showVelocityVectors })),
}));
