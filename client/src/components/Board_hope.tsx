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
import { start } from "repl";
import { after } from "next/server";

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
const safeMap: Record<string, number[]> = {
  Red: [100, 101, 102, 103, 104, 105],
  Yellow: [200, 201, 202, 203, 204, 205],
  Blue: [300, 301, 302, 303, 304, 305],
  Green: [400, 401, 402, 403, 404, 405]
};

//---PATH FUNCTION------------------------------------------------------------------------------------------
const pawnColors = ['Red', 'Yellow', 'Green', 'Blue'];
function getAnimationPath (lastMoveInfo: any, pawns: {id: string, tileId: number, color: string}[]) {
  const color = lastMoveInfo?.['color'] != null ? lastMoveInfo['color'] : 'Red';
  let startIdx = lastMoveInfo?.['before_board_idx'] != null ? lastMoveInfo['before_board_idx'] : null;
  let endIdx = lastMoveInfo?.['after_board_idx'] != null ? lastMoveInfo['after_board_idx'] : null;
  let dice = lastMoveInfo?.['dice'] != null ? lastMoveInfo['dice'] : null;
  let afterTilePosition = lastMoveInfo?.['after_tile_position'] != null ? lastMoveInfo['after_tile_position'] : 0;
  let beforeTilePosition = lastMoveInfo?.['before_tile_position'] != null ? lastMoveInfo['before_tile_position'] : 0;
  let pawn = lastMoveInfo?.['chosen'] != null ? lastMoveInfo['chosen'] : null;
  let pawn_id = Number(pawn?.split('_')[1] ?? 0);

  let after_pawn_tileId = Number(afterTilePosition);
  let before_pawn_tileId = Number(beforeTilePosition);
  if (after_pawn_tileId > 49 && after_pawn_tileId < 56) {
    if (before_pawn_tileId < 50) {
      if (color === 'Red') endIdx = 49;
      if (color === 'Yellow') endIdx = 10;
      if (color === 'Blue') endIdx = 23;
      if (color === 'Green') endIdx = 36;
      return [endIdx, safeMap[color][after_pawn_tileId - 50]]; 
    }
    return [safeMap[color][before_pawn_tileId - 50], safeMap[color][after_pawn_tileId - 50]];
  }

  let path: number[] = [];
  // start part  
  if (startIdx == null && endIdx == null) {
    if (color === 'Red') return [1000 + pawn_id, 99];
    if (color === 'Yellow') return[2000 + pawn_id, 199];
    if (color === 'Blue') return[3000 + pawn_id, 299];
    if (color === 'Green') return[4000 + pawn_id, 399];
  }

  let idx = startIdx ?? 0;
  if (startIdx == null && endIdx != null) {
    if (color === 'Red') path.push(99);
    if (color === 'Yellow') path.push(199);
    if (color === 'Blue') path.push(299);
    if (color === 'Green') path.push(399);
    
    if (color === 'Red') idx = 0;
    if (color === 'Yellow') idx = 13; 
    if (color === 'Blue') idx = 26;
    if (color === 'Green') idx = 39;
    dice -= 1;
  }

  path.push(idx);
  for (let i = 0; i < dice; i++) {
    idx = (idx + 1) % 52;
    path.push(idx);
  }

  while (endIdx != idx) {
    if ((color == 'Green' && idx == 4) ||
        (color == 'Red' && idx == 17) ||
        (color == 'Yellow' && idx == 30) ||
        (color == 'Blue' && idx == 43)) {
      idx += 12;
      idx = idx % 52;
    } else {    
      idx += 4;
      idx = idx % 52;
    }
    path.push(idx);
  }

  return path;

}


function getTileMotionPath(tileIds: number[]) {
  return tileIds.map(id => {
    const tile = baseTilesInit.find(t => t.id === id);
    if (tile == undefined) { 
      return {x: 0, y:0, rotate:0}
    }
    return {
      x: `${tile?.x - 3.4}%`,
      y: `${tile?.y - 3.4}%`,
      rotate: tile?.degree ?? 0
    };
  });
}

function getAnimationProps (path: number[], delay: number) {
  const motionPath = getTileMotionPath(path); // example tile IDs
  const xPath = motionPath.map(p => p.x);
  const yPath = motionPath.map(p => p.y);
  const rotatePath = motionPath.map(p => p.rotate);
  
  const stepCount = motionPath.length;
  const times = Array.from({ length: stepCount }, (_, i) => i / (stepCount - 1));
  const durationPerStep = 0.3;
  const totalDuration = durationPerStep * (stepCount - 1);
  return {xPath: xPath, yPath: yPath, rotatePath: rotatePath, times: times, totalDuration: totalDuration, delay: delay}
}

function Board() {

  //---AUDIO Instantiations-------------------------------------------------------
  const moveAudioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    // Initialize audio once. Public assets are served from root in Next.js
    try {
      moveAudioRef.current = new Audio('/plane-fly.m4a');
      moveAudioRef.current.preload = 'auto';
      moveAudioRef.current.load();
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
      const newPawns: { id: string; tileId: number; color: string }[] = [];
      
      state.players.forEach((pl: any) => {
        const color = pl.color;
        pl.pawns.forEach((pw: any) => {
          let tileId = pw.tile_position ?? 0;
            
          // check if they are in base (tileId = 0 ) and if in safe area -?
          if (tileId == -2) {
            if (color == 'Red') tileId = 1000 + Number(pw.id.split('_')[1]);            
            if (color == 'Yellow') tileId = 2000 + Number(pw.id.split('_')[1]);            
            if (color == 'Blue') tileId = 3000 + Number(pw.id.split('_')[1]);            
            if (color == 'Green') tileId = 4000 + Number(pw.id.split('_')[1]);         
          }

          newPawns.push({ id: pw.id, tileId: tileId, color: color });
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
          //play audio when pawn positions changed
          if (moveAudioRef.current) {
            moveAudioRef.current.currentTime = 0;
            
            moveAudioRef.current.play().catch((err) => console.error('Audio play failed:', err));
          }
        }
        setPawns(newPawns);
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
  const [pawns, setPawns] = useState<{ id: string; tileId: number; color: string }[]>([]);
  const pawnColors = ["#fbbf24", "#ef4444", "#10b981", "#3b82f6"];  
  const [animationTargets, setAnimationTargets] = useState<Record<string, any>>({});
  
  function getPlaneSrcFromHex(hexOrColor?: string) {
    // Return an image src string for a given player color hex or name
    const pRed = (planeRed as any)?.src || planeRed;
    const pBlue = (planeBlue as any)?.src || planeBlue;
    const pGreen = (planeGreen as any)?.src || planeGreen;
    const pYellow = (planeYellow as any)?.src || planeYellow;

    const val = (hexOrColor || '').toString().toLowerCase();

    const pc = pawnColors.findIndex(c => c.toLowerCase() === val);
    if (pc === 0 || val == 'yellow') return pYellow;
    if (pc === 1 || val == 'red') return pRed;
    if (pc === 2 || val == 'green') return pGreen;
    if (pc === 3 || val == 'blue') return pBlue;

    return pBlue;
  }
  
  // Tile variables and useStates
  const [tiles, setTiles] = useState<{ id: number; x: number; y: number; size: number; color: string, degree: number }[]>(baseTilesInit);

  // Server retrieval variables and useStates
  const [serverState, setServerState] = useState<any | null>(null);
  const [lastMoveInfo, setLastMoveInfo] = useState<any | null>(null);
  const [animeDelay, setAnimeDelay] = useState<number>(0);
  
  const playerColorMap: Record<string, string> = {
    Red: '#ef4444',
    Blue: '#3b82f6',
    Green: '#10b981',
    Yellow: '#fbbf24'
  };  
  useEffect(() => {
    // Only run if lastMoveInfo has been updated with a valid move
    if (lastMoveInfo && lastMoveInfo['chosen']) {
      
      const pawnIdToAnimate = lastMoveInfo['chosen'];
      const path = getAnimationPath(lastMoveInfo, pawns);
      const destinationProps = getAnimationProps(path, animeDelay);

      if (destinationProps) {
        // ✅ Set the animation target for the pawn that just moved
        setAnimationTargets(prev => ({
          ...prev,
          [pawnIdToAnimate]: destinationProps
        }));
      }

      setAnimationTargets(prev => {
        const updated = { ...prev };
        pawns.forEach(pawn => {
          if (pawn.tileId >= 1000 && animationTargets[pawn.id] != null) {
            delete updated[pawn.id]; // Remove animation target
            
            // play blast sound
            try {
              moveAudioRef.current = new Audio('/missile-firing.mp3');
              moveAudioRef.current.play().catch((err) => console.error('Audio play failed:', err));
              moveAudioRef.current = new Audio('/plane-fly.m4a');
            } catch (e) {
              console.warn('Audio init failed', e);
            }
          }
        });
        return updated;
      });
    }
  }, [lastMoveInfo]);

  useEffect(() => {
  if (!serverState?.players) return;

  const winner = serverState.players.find((p : {color: string; finished_pawns: number}) => p.finished_pawns === 4);
    // play landing sound
    if (winner) {
      try {
        moveAudioRef.current = new Audio('/pilot-announcement.mp3');
        moveAudioRef.current.play().catch((err) => console.error('Audio play failed:', err));
        moveAudioRef.current = new Audio('/plane-fly.m4a');
      } catch (e) {
        console.warn('Audio init failed', e);
      }
      alert(`🎉 ${winner.color} wins the game!`);
    }
  }, [serverState?.players]);




  //----------BOARD RETURN HTML---------------------------------------
  return (
    <div>
    <div
      className={`${styles.imageContainer} relative rounded-lg overflow-hidden`}
      style={{
        width: "min(85vmin, 900px)",
        height: "min(85vmin, 900px)",
        backgroundImage: `url(${(aeroplane as any)?.src ?? (aeroplane as any)})`,
        backgroundSize: 'contain',
        backgroundRepeat: 'no-repeat',
        backgroundPosition: 'center',
        alignItems: 'center'
      }}
    >
      
      {pawns.map((pawn) => {
        const anim = animationTargets[pawn.id];
        const tile = tiles?.find(t => t.id === pawn.tileId) ?? {x: 0, y: 0, degree: 0};

        return (
          <motion.img
            key={pawn.id}
            src={getPlaneSrcFromHex(pawn.color)}
            alt={`Pawn ${pawn.id}`}
            title={`${pawn.id}`}
            className={styles.planeImage}
            initial={{
              left: anim?.xPath?.[0] ?? `${tile?.x - 3.4}%`,
              top: anim?.yPath?.[0] ?? `${tile?.y - 3.4}%`,
              rotate: anim?.rotatePath?.[0] ?? tile?.degree ?? 0
            }}
            animate={{
              left: anim?.xPath ?? `${tile?.x - 3.4}%`,
              top: anim?.yPath ?? `${tile?.y - 3.4}%`,
              rotate: anim?.rotatePath ?? [tile?.degree ?? 0]
            }}
            transition={{
              duration: anim?.totalDuration ?? 1,
              ease: "easeInOut",
              times: anim?.times ?? undefined,
              delay: anim?.delay ?? undefined
            }}
            onClick={async () => {
              const sp_id = pawn.id; // your pawn id
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
              width: `6.8%`,
              height: `6.8%`,
              borderRadius: 4,
              transformOrigin: "50% 50%"
            }}
          />
        );
        
      })}

      {serverState?.players?.map((player: {
        color: string;
        finished_pawns: number;
      }) => {
        let style = {}
        if (player.color === 'Red') {
          style = { position: 'absolute', color: 'red', left: `45%`, top: `55%`, fontSize: 17}
        } else if (player.color === 'Yellow') {
          style = { position: 'absolute', color: 'orange', left: `42.5%`, top: `43.5%`, fontSize: 17}
        } else if (player.color === 'Blue') {
          style = { position: 'absolute', color: 'blue', left: `53.5%`, top: `41.5%`, fontSize: 17}
        } else if (player.color === 'Green') {
          style = { position: 'absolute', color: 'green', left: `56.25%`, top: `52.5%`, fontSize: 17}
        }          
        return (
          <div style={style}>
            <b>{player.finished_pawns}</b>
          </div>
        );
      })}
    
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
              const newPawns: { id: string; tileId: number; color: string }[] = [];

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
                      if (match.id == 0) match.id = -2;
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
      <button onClick={() => moveAudioRef.current?.play()}>Play Audio</button>

      <label htmlFor="delaySlider" style={{fontSize: 18}}>
        Delay: {animeDelay.toFixed(2)}s
      </label>
      <input
        id="delaySlider"
        type="range"
        min="0"
        max="5"
        step="0.1"
        value={animeDelay}
        onChange={(e) => setAnimeDelay(parseFloat(e.target.value))}
        style={{ width: '100%' }}
      />
       
    </div>

   </div>

  );
}

export default Board;