import React, { useState, useRef } from "react";
import styles from '@/styles/Board.module.css';

// BOARD IMG import
import aeroplane from '../../img/aeroplane.png';
import planeRed from '../../img/plane_red_b.png';
import planeBlue from '../../img/plane_blue_b.png';
import planeGreen from '../../img/plane_green_b.png';
import planeYellow from '../../img/plane_yellow_b.png';
import { motion } from "framer-motion"

import { useEffect } from "react";

//---BASE TILE FUNCTIONS-----------------------------------------------------------------------------------
const defaultTileSize = 3.2;
const tileColors = [
  "#f87171", // red
  "#34d399", // green
  "#60a5fa", // blue
  "#fbbf24", // yellow
]
const baseTilesInit: { id: number; x: number; y: number; size: number; color: string; degree: number }[] = [
    // Red Base
    { id: 1001, x: 5.9, y: 82.3, size: defaultTileSize, color: tileColors[0], degree: 45 },
    { id: 1002, x: 5.9, y: 94.2, size: defaultTileSize, color: tileColors[0], degree: 45 },
    { id: 1003, x: 17.6, y: 82.3, size: defaultTileSize, color: tileColors[0], degree: 45 },
    { id: 1004, x: 17.6, y: 94.2, size: defaultTileSize, color: tileColors[0], degree: 45 },
    // Yello Base
    { id: 2001, x: 5.9, y: 5.9, size: defaultTileSize, color: tileColors[3], degree: 135 },
    { id: 2002, x: 17.6, y: 5.9, size: defaultTileSize, color: tileColors[3], degree: 135 },
    { id: 2003, x: 5.9, y: 17.6, size: defaultTileSize, color: tileColors[3], degree: 135 },
    { id: 2004, x: 17.6, y: 17.6, size: defaultTileSize, color: tileColors[3], degree: 135 },
    // Blue Base
    { id: 3001, x: 82.3, y: 5.9, size: defaultTileSize, color: tileColors[2], degree: -135 },
    { id: 3002, x: 94.2, y: 5.9, size: defaultTileSize, color: tileColors[2], degree: -135 },
    { id: 3003, x: 82.3, y: 17.6, size: defaultTileSize, color: tileColors[2], degree: -135 },
    { id: 3004, x: 94.2, y: 17.6, size: defaultTileSize, color: tileColors[2], degree: -135 },
    // Green Base
    { id: 4001, x: 82.3, y: 82.3, size: defaultTileSize, color: tileColors[1], degree: -45 },
    { id: 4002, x: 82.3, y: 94.2, size: defaultTileSize, color: tileColors[1], degree: -45 },
    { id: 4003, x: 94.2, y: 82.3, size: defaultTileSize, color: tileColors[1], degree: -45 },
    { id: 4004, x: 94.2, y: 94.2, size: defaultTileSize, color: tileColors[1], degree: -45 },

    // Exit Start Area Tiles
    { id: 99, x: 27.5, y: 96.5, size: defaultTileSize, color: tileColors[0], degree: 45 },
    { id: 199, x: 3.5, y: 27.5, size: defaultTileSize, color: tileColors[3], degree: 135 },
    { id: 299, x: 72.5, y: 3.5, size: defaultTileSize, color: tileColors[2], degree: 180 },
    { id: 399, x: 96.5, y: 73.0, size: defaultTileSize, color: tileColors[1], degree: -45 },

    // Red Safe Area Arrow Tiles
    { id: 105, x: 50.2, y: 55.2, size: defaultTileSize, color: tileColors[0], degree: 0 },
    { id: 104, x: 50.2, y: 61.7, size: defaultTileSize, color: tileColors[0], degree: 0 },
    { id: 103, x: 50.2, y: 68.3, size: defaultTileSize, color: tileColors[0], degree: 0 },
    { id: 102, x: 50.2, y: 74.4, size: defaultTileSize, color: tileColors[0], degree: 0 },
    { id: 101, x: 50.2, y: 81.1, size: defaultTileSize, color: tileColors[0], degree: 0 },
    { id: 100, x: 50.2, y: 87.8, size: defaultTileSize, color: tileColors[0], degree: 0 },
    // Yellow Safe Area Arrow Tiles
    { id: 200, x: 12.6, y: 49.9, size: defaultTileSize, color: tileColors[3], degree: 90 },
    { id: 201, x: 19.1, y: 49.9, size: defaultTileSize, color: tileColors[3], degree: 90 },
    { id: 202, x: 25.7, y: 49.9, size: defaultTileSize, color: tileColors[3], degree: 90 },
    { id: 203, x: 31.8, y: 49.9, size: defaultTileSize, color: tileColors[3], degree: 90 },
    { id: 204, x: 38.5, y: 49.9, size: defaultTileSize, color: tileColors[3], degree: 90 },
    { id: 205, x: 44.8, y: 49.9, size: defaultTileSize, color: tileColors[3], degree: 90 },
    // Blue Safe Area Arrow Tiles
    { id: 300, x: 50.2, y: 12.3, size: defaultTileSize, color: tileColors[2], degree: 180 },
    { id: 301, x: 50.2, y: 18.8, size: defaultTileSize, color: tileColors[2], degree: 180 },
    { id: 302, x: 50.2, y: 25.4, size: defaultTileSize, color: tileColors[2], degree: 180 },
    { id: 303, x: 50.2, y: 31.5, size: defaultTileSize, color: tileColors[2], degree: 180 },
    { id: 304, x: 50.2, y: 38.2, size: defaultTileSize, color: tileColors[2], degree: 180 },
    { id: 305, x: 50.2, y: 44.9, size: defaultTileSize, color: tileColors[2], degree: 180 },
    // Green Safe Area Arrow Tiles
    { id: 405, x: 55.2, y: 49.9, size: defaultTileSize, color: tileColors[1], degree: -90 },
    { id: 404, x: 61.9, y: 49.9, size: defaultTileSize, color: tileColors[1], degree: -90 },
    { id: 403, x: 68.0, y: 49.9, size: defaultTileSize, color: tileColors[1], degree: -90 },
    { id: 402, x: 74.6, y: 49.9, size: defaultTileSize, color: tileColors[1], degree: -90 },
    { id: 401, x: 81.1, y: 49.9, size: defaultTileSize, color: tileColors[1], degree: -90 },
    { id: 400, x: 87.6, y: 49.9, size: defaultTileSize, color: tileColors[1], degree: -90 },

    { id: 0, x: 31.8, y: 91.8, size: defaultTileSize, color: '#94a3b8', degree: -45 },
    { id: 1, x: 29.4, y: 85.2, size: defaultTileSize, color: '#94a3b8', degree: 0 },
    { id: 2, x: 29.4, y: 79.4, size: defaultTileSize, color: '#94a3b8', degree: 0 },
    { id: 3, x: 31.7, y: 73.0, size: defaultTileSize, color: '#94a3b8', degree: -45 },
    { id: 4, x: 27.4, y: 68.3, size: defaultTileSize, color: '#94a3b8', degree: -90 },
    { id: 5, x: 20.5, y: 70.6, size: defaultTileSize, color: '#94a3b8', degree: -90 },
    { id: 6, x: 14.8, y: 70.6, size: defaultTileSize, color: '#94a3b8', degree: -90 },
    { id: 7, x: 8.1, y: 68.3, size: defaultTileSize, color: '#94a3b8', degree: -45 },
    { id: 8, x: 5.9, y: 61.9, size: defaultTileSize, color: '#94a3b8', degree: 0 },
    { id: 9, x: 5.9, y: 55.9, size: defaultTileSize, color: '#94a3b8', degree: 0 },
    { id: 10, x: 5.9, y: 49.9, size: defaultTileSize, color: '#94a3b8', degree: 0 },
    { id: 11, x: 5.9, y: 43.9, size: defaultTileSize, color: '#94a3b8', degree: 0 },
    { id: 12, x: 5.9, y: 37.9, size: defaultTileSize, color: '#94a3b8', degree: 0 },
    { id: 13, x: 8.1, y: 31.8, size: defaultTileSize, color: '#94a3b8', degree: 45 },
    { id: 14, x: 14.8, y: 29.4, size: defaultTileSize, color: '#94a3b8', degree: 90 },
    { id: 15, x: 20.5, y: 29.4, size: defaultTileSize, color: '#94a3b8', degree: 90 },
    { id: 16, x: 27.4, y: 31.7, size: defaultTileSize, color: '#94a3b8', degree: 45 },
    { id: 17, x: 31.7, y: 27.4, size: defaultTileSize, color: '#94a3b8', degree: 45 },
    { id: 18, x: 29.4, y: 20.5, size: defaultTileSize, color: '#94a3b8', degree: 0 },
    { id: 19, x: 29.4, y: 14.8, size: defaultTileSize, color: '#94a3b8', degree: 0 },
    { id: 20, x: 31.8, y: 8.1, size: defaultTileSize, color: '#94a3b8', degree: 45 },
    { id: 21, x: 38.2, y: 5.9, size: defaultTileSize, color: '#94a3b8', degree: 90 },
    { id: 22, x: 44.2, y: 5.9, size: defaultTileSize, color: '#94a3b8', degree: 90 },
    { id: 23, x: 50.2, y: 5.9, size: defaultTileSize, color: '#94a3b8', degree: 90 },
    { id: 24, x: 56.2, y: 5.9, size: defaultTileSize, color: '#94a3b8', degree: 90 },
    { id: 25, x: 62.2, y: 5.9, size: defaultTileSize, color: '#94a3b8', degree: 90 },
    { id: 26, x: 68.7, y: 8.1, size: defaultTileSize, color: '#94a3b8', degree: 135 },
    { id: 27, x: 70.6, y: 14.8, size: defaultTileSize, color: '#94a3b8', degree: 180 },
    { id: 28, x: 70.6, y: 20.5, size: defaultTileSize, color: '#94a3b8', degree: 180 },
    { id: 29, x: 68.3, y: 27.4, size: defaultTileSize, color: '#94a3b8', degree: 135 },
    { id: 30, x: 73.0, y: 31.7, size: defaultTileSize, color: '#94a3b8', degree: 135 },
    { id: 31, x: 79.4, y: 29.4, size: defaultTileSize, color: '#94a3b8', degree: 90 },
    { id: 32, x: 85.2, y: 29.4, size: defaultTileSize, color: '#94a3b8', degree: 90 },
    { id: 33, x: 91.8, y: 31.7, size: defaultTileSize, color: '#94a3b8', degree: 135 },
    { id: 34, x: 94.2, y: 37.9, size: defaultTileSize, color: '#94a3b8', degree: 180 },
    { id: 35, x: 94.2, y: 43.9, size: defaultTileSize, color: '#94a3b8', degree: 180 },
    { id: 36, x: 94.2, y: 49.9, size: defaultTileSize, color: '#94a3b8', degree: 180 },
    { id: 37, x: 94.2, y: 55.9, size: defaultTileSize, color: '#94a3b8', degree: 180 },
    { id: 38, x: 94.2, y: 61.9, size: defaultTileSize, color: '#94a3b8', degree: 180 },
    { id: 39, x: 91.8, y: 68.3, size: defaultTileSize, color: '#94a3b8', degree: 225 },
    { id: 40, x: 85.2, y: 70.6, size: defaultTileSize, color: '#94a3b8', degree: 270 },
    { id: 41, x: 79.4, y: 70.6, size: defaultTileSize, color: '#94a3b8', degree: 270 },
    { id: 42, x: 73.0, y: 68.3, size: defaultTileSize, color: '#94a3b8', degree: 225 },
    { id: 43, x: 68.3, y: 73.0, size: defaultTileSize, color: '#94a3b8', degree: 225 },
    { id: 44, x: 70.6, y: 79.4, size: defaultTileSize, color: '#94a3b8', degree: 180 },
    { id: 45, x: 70.6, y: 85.2, size: defaultTileSize, color: '#94a3b8', degree: 180 },
    { id: 46, x: 68.7, y: 91.8, size: defaultTileSize, color: '#94a3b8', degree: 225 },
    { id: 47, x: 62.2, y: 94.2, size: defaultTileSize, color: '#94a3b8', degree: 270 },
    { id: 48, x: 56.2, y: 94.2, size: defaultTileSize, color: '#94a3b8', degree: 270 },
    { id: 49, x: 50.2, y: 94.2, size: defaultTileSize, color: '#94a3b8', degree: 270 },
    { id: 50, x: 44.2, y: 94.2, size: defaultTileSize, color: '#94a3b8', degree: 270 },
    { id: 51, x: 38.2, y: 94.2, size: defaultTileSize, color: '#94a3b8', degree: 270 },  
];
const baseTileIds: Record<string, number[]> = {
    Red: [1001, 1002, 1003, 1004],
    Yellow: [2001, 2002, 2003, 2004],
    Blue: [3001, 3002, 3003, 3004],
    Green: [4001, 4002, 4003, 4004]
};

//---PATH FUNCTION------------------------------------------------------------------------------------------
const pawnColors = ['Red', 'Yellow', 'Green', 'Blue'];
function getAnimationProps (lastMoveInfo: any) {
  const color = lastMoveInfo?.['color'] != null ? lastMoveInfo['color'] : 'Red';
  let startIdx = lastMoveInfo?.['before_board_idx'] != null ? lastMoveInfo['before_board_idx'] : null;
  let endIdx = lastMoveInfo?.['after_board_idx'] != null ? lastMoveInfo['after_board_idx'] : null;
  let dice = lastMoveInfo?.['dice'] != null ? lastMoveInfo['dice'] : null;

  if (dice == null) {
    return;
  }
  if (startIdx == null || startIdx == -1) {
    if (color === 'Red') {
      startIdx = 99;
    } else if (color === 'Yellow') {
      startIdx = 199;
    } else if (color === 'Blue') {
      startIdx = 299;
    } else if (color === 'Green') {
      startIdx = 399
    }
  }

  if (endIdx == null || endIdx == -1) {
    if (color === 'Red') {
      return [99];
    } else if (color === 'Yellow') {
      return [199];
    } else if (color === 'Blue') {
      return [299];
    } else if (color === 'Green') {
      return [399];
    }
  }
   
  const path = [];
  let idx = startIdx;
  for(let i = 0; i < dice + 1; i++) {
    path.push(idx);
    if (idx > 55) {
      if (color === 'Red') {
        idx = 0;
      } else if (color === 'Yellow') {
        idx = 13;
      } else if (color === 'Blue') {
        idx = 26;
      } else if (color === 'Green') {
        idx = 39;
      }
      continue;
    }
    idx += 1;
    idx = idx % 52;
  }

  idx -= 1;
  while (endIdx > idx) {
    if ((color == 'Green' && idx == 4) ||
        (color == 'Red' && idx == 17) ||
        (color == 'Yellow' && idx == 30) ||
        (color == 'Blue' && idx == 43)) {
      idx += 12;
    } else {    
      idx += 4;
    }
    path.push(idx);
  }   
  
  return path;
}

function Board() {

  //---AUDIO Instantiations-------------------------------------------------------
  const moveAudioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    // Initialize audio once. Public assets are served from root in Next.js
    try {
      moveAudioRef.current = new Audio('/move-self.mp3');
      moveAudioRef.current.volume = 0.9;
    } catch (e) {
      console.warn('Audio init failed', e);
    }
  }, []);

  //---REFRESH Methods-------------------------------------------------------
  const refreshBoard = async () => {
    try {
      const gid = localStorage.getItem('game_id');
      if (!gid) return alert('No game in progress');

      // fetch and store data
      const res = await fetch(`http://localhost:8080/api/play/refresh`);
      const data = await res.json();
      const state = data.state;
      // update vars
      setServerState(state);
      setLastMoveInfo(data.last_move ?? null);
      
      // Rebuild pawns from server snapshot
      const newPawns: { id: number; tileId: number; color: string }[] = [];
      state.players.forEach((pl: any) => {
        const color = pl.color;
        pl.pawns.forEach((pw: any) => {
          let tileId = pw.tile_position ?? baseTileIds[color][0];
          newPawns.push({ id: pw.id, tileId, color });
        });
      });

      // Only update pawns if something actually changed to avoid noisy re-renders
      try {
        const prevPawns = pawns || [];
        const pawnsChanged = (() => {
          if (prevPawns.length !== newPawns.length) return true;
          const map = new Map(prevPawns.map((p) => [String(p.id), p.tileId]));
          for (const np of newPawns) {
            if (map.get(String(np.id)) !== np.tileId) return true;
          }
          return false;
        })();

        if (pawnsChanged) {
          setPawns(newPawns);
          // play audio when pawn positions changed
          if (moveAudioRef.current) {
            moveAudioRef.current.currentTime = 0;
            moveAudioRef.current.play().catch((err) => console.error('Audio play failed:', err));
          }
        }

      } catch (e) {
        alert('pawn compare or last move compare failed' + e);
      }
    } catch {}
  };
  
  useEffect(() => {
    // Prefer Server-Sent Events so we only refresh when server emits a move
    const cleanup: Array<() => void> = [];
    let es: EventSource | null = null;
    const streamUrl = 'http://localhost:8080/api/play/stream';
    if (typeof window !== 'undefined' && 'EventSource' in window) {
        try {
          es = new EventSource(streamUrl);
          es.onmessage = (ev) => {
            try {
              const payload = JSON.parse(ev.data);
              if (payload && payload.type === 'move') {
                // only refresh on explicit move events
                refreshBoard();
              }
            } catch {}
          };
        } catch (e) {
          alert('SSE init failed, falling back to polling' + e);
        }
    } 
    return () => {
      if (es) es.close();
      cleanup.forEach((fn) => fn());
    };
  }, []);

  //-------VARIABLE AND USESTATES INSTANTIATION--------------------------------------------------------------------
  // Pawn variables and useStates
  const [pawns, setPawns] = useState<{ id: number; tileId: number; color: string }[]>([]);
  const pawnColors = ["#fbbf24", "#ef4444", "#10b981", "#3b82f6"];  

  function getPlaneSrcFromHex(hexOrColor?: string) {
    // Return an image src string for a given player color hex or name
    const pRed = (planeRed as any)?.src || planeRed;
    const pBlue = (planeBlue as any)?.src || planeBlue;
    const pGreen = (planeGreen as any)?.src || planeGreen;
    const pYellow = (planeYellow as any)?.src || planeYellow;

    const val = (hexOrColor || '').toString().toLowerCase();
    const pc = pawnColors.findIndex(c => c.toLowerCase() === val);
    if (pc === 0) return pYellow;
    if (pc === 1) return pRed;
    if (pc === 2) return pGreen;
    if (pc === 3) return pBlue;
    return pBlue;
  }
  
  // Tile variables and useStates
  const [tiles, setTiles] = useState<{ id: number; x: number; y: number; size: number; color: string, degree: number }[]>(baseTilesInit);
  const safeMap: Record<string, number[]> = {
    Red: [100, 101, 102, 103, 104, 105],
    Yellow: [200, 201, 202, 203, 204, 205],
    Blue: [300, 301, 302, 303, 304, 305],
    Green: [400, 401, 402, 403, 404, 405]
  };

  // Server retrieval variables and useStates
  const [serverState, setServerState] = useState<any | null>(null);
  const [lastMoveInfo, setLastMoveInfo] = useState<any | null>(null);
  const [availableMoves, setAvailableMoves] = useState<any | null>(null);
  const playerColorMap: Record<string, string> = {
    Red: '#ef4444',
    Blue: '#3b82f6',
    Green: '#10b981',
    Yellow: '#fbbf24'
  };  


  //----------BOARD RETURN HTML---------------------------------------
  return (
    <div>
       <div
          className={`${styles.imageContainer} relative rounded-lg overflow-hidden`}
          style={{
            width: "min(80vmin, 800px)",
            height: "min(80vmin, 800px)",
            backgroundImage: `url(${(aeroplane as any)?.src ?? (aeroplane as any)})`,
            backgroundSize: 'contain',
            backgroundRepeat: 'no-repeat',
            backgroundPosition: 'center',
            alignItems: 'center'
          }}
        >
          {(() => {

            // tileMap that stores pawn locations of all players (store and create)
            const serverTileMap: Record<number, { id: string; color: string; tile_position: number | null }[]> = {};
            if (serverState) {
              serverState.players.forEach((pl: any) => {
                const color = playerColorMap[pl.color] || '#000';
                
                pl.pawns.forEach((pw: any) => {
                  let target = null as number | null;
                  // prefer explicit board index when provided for main-track positions
                  const tilePos = pw.tile_position ?? null;
                  // server uses -1 to indicate a special 'start' marker; map Red's start marker to UI tile 99
                  if (tilePos === -1) {
                    if (pl.color === 'Red') {
                      target = 99;
                    } else if (pl.color === 'Yellow') {
                      target = 199;
                    } else if (pl.color === 'Blue') {
                      target = 299;
                    } else if (pl.color === 'Green') {
                      target = 399
                    }
                  } else if (tilePos !== null && tilePos >= 50) {
                    // pawn is in safe area; map to color-specific safe UI tiles
                    const safeIdx = tilePos - 50; // 0..6 maybe; we expect 0..5 for six safe slots
                    const safes = safeMap[pl.color] || [];
                    if (safeIdx >= 0 && safeIdx < safes.length) {
                      target = safes[safeIdx];
                    }
                  } else if (pw.board_idx !== null && pw.board_idx !== undefined) {
                    target = pw.board_idx;
                  } else if (tilePos !== null && tilePos > 0) {
                    target = tilePos;
                  } else {
                    // start area -> pick base mapping
                    const parts = (pw.id || '').split('_');
                    const idx = parts.length > 1 ? parseInt(parts[1], 10) - 1 : 0;
                    const bases = baseTileIds[pl.color];
                    if (bases && bases[idx] !== undefined) target = bases[idx];
                  }
                  if (target !== null && target !== undefined) {
                    if (!serverTileMap[target]) serverTileMap[target] = [];
                    serverTileMap[target].push({ id: pw.id, color, tile_position: pw.tile_position ?? null });
                  }
                });
              });
            }

            // place tiles with pawns within them
            return tiles.map((t) => {
              const serverPawns = serverTileMap[t.id] || [];
              return (
                <div
                  key={t.id}
                  className={`${styles.tileCircle} `}
                  style={{
                    left: `${t.x}%`,
                    top: `${t.y}%`,
                    width: `${t.size}vmin`,
                    height: `${t.size}vmin`,
                    transform: 'translate(-50%, -50%)'
                  }}
                  title={`Tile ${t.id}`}
                >

                  {/* Pawn Mapping */}
                  {serverPawns.map((sp, i) => {
                    const isAvailable = availableMoves?.some((m:any) => m.id === sp.id);
                    // Check if the current turn is a human player's turn before allowing a click
                 
                    return (
                      <div
                        key={sp.id + '_' + i}
                        onClick={async () => {
                          const sp_id = sp.id; // your pawn id
                          try {
                            const gid = localStorage.getItem('game_id');
                            if (!gid) return alert('No game in progress');
                            const res = await fetch("http://localhost:8080/api/play/choose", {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({ pawn_id: sp_id, game_id: gid })
                            });
                            if (res.ok) {
                              // refresh immediately and play audio
                              await refreshBoard();
                            } else {
                              alert('choose returned' + res.status);
                            }

                          } catch {}
                        }}
                        style={{
                          position: "absolute",
                          right: -6.8,
                          top: -5,
                          width: 35,
                          height: 35,
                          borderRadius: 4,
                          background: 'transparent',
                          cursor: isAvailable ? "pointer" : "default" // Adjust cursor for clarity
                        }}
                        title={`${sp.id} pos:${sp.tile_position}`}
                      >
                        {/* THE ACTUAL PAWN */}
                        {(() => {
                          const angleForTile = t.degree;
                          return (
                            <img src={getPlaneSrcFromHex(sp.color)} alt="pawn" className={styles.planeImage} style={{ width: '100%', height: '100%', display: 'block', transform: `rotate(${angleForTile}deg)`, transformOrigin: '50% 50%', filter: isAvailable ? 'drop-shadow(0 0 6px yellow)' : `drop-shadow(0 0 6px white})` }} />
                          );
                        })()}
                      </div>
                    );
                  })}
                </div>
              );
            });
          })()}
        </div>

       <div className={styles.controls}>

         {/* Start Game Button */}
          <button onClick={async () => {
            try {
              const res = await fetch('http://localhost:8080/api/play/start', { method: 'POST' });
              const data = await res.json();
              const gid = data.game_id;
              // store game id in localStorage for demo; in a real app use context or state
              localStorage.setItem('game_id', gid);

              // map server pawn positions to on-screen pawns
              const state = data.state;
              setServerState(state);
              const newPawns: { id: number; tileId: number; color: string }[] = [];

              state.players.forEach((pl: any, pi: number) => {
                const color = playerColorMap[pl.color] || pawnColors[pi % pawnColors.length];
                pl.pawns.forEach((pw: any) => {
                  if (pw.board_idx !== null && pw.board_idx !== undefined) {
                    // server provided board index (main track index); try to match a tile with same id
                    const match = tiles.find(t => t.id === pw.board_idx);
                    if (match) {
                      newPawns.push({ id: pw.id, tileId: match.id, color });
                      return;
                    }
                  }

                // pawn in start area (tile_position == 0) -> place in corresponding base slot
                // pawn id format: Color_<n>
                const parts = (pw.id || '').split('_');
                const idx = parts.length > 1 ? parseInt(parts[1], 10) - 1 : 0;
                const bases = baseTileIds[pl.color];
                if (bases && bases[idx] !== undefined) {
                    const baseTileId = bases[idx];
                    const match = tiles.find(t => t.id === baseTileId);
                    if (match) {
                    newPawns.push({ id: pw.id, tileId: match.id, color });
                    }
                }
                });
              });
              setPawns(newPawns);
            } catch (err) {
              alert("Start Game Failed");
            }
          }}>Start Game</button>
            
        <div>
        {lastMoveInfo?.['before_board_idx'] != null ? lastMoveInfo['before_board_idx'] : 'Before is None'}
        </div>
        <div>
        {lastMoveInfo?.['after_board_idx'] != null ? lastMoveInfo['after_board_idx'] : 'After is None'}
        </div>
        <div>{getAnimationProps(lastMoveInfo)}</div>

     
      </div>
  </div>
       
  );
}

export default Board;