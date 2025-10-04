import React, { useState, useRef, useEffect } from "react";
import styles from '@/styles/Board.module.css';

// BOARD IMG import
import aeroplane from '../../img/aeroplane.png';
import planeRed from '../../img/plane_red_b.png';
import planeBlue from '../../img/plane_blue_b.png';
import planeGreen from '../../img/plane_green_b.png';
import planeYellow from '../../img/plane_yellow_b.png';
import { motion, useAnimation } from "framer-motion"
import { pre } from "framer-motion/client";

// State definitions
// Type definitions
interface PawnState {
  board_idx: number | null;
  id: string;
  tile_position: number;
}

interface Player {
  color: string;
  pawns: PawnState[];
}

interface GameState {
  current_player: number;
  players: Player[];
}

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
function getTileById(key: string, id: number) {
  const tile = baseTilesInit.find(t => t.id === id);  
  return {
    key: key,
    currentTileId: tile?.id ?? -1,
    x: tile?.x ?? -1,
    y: tile?.y ?? -1,
    color: tile?.color ?? "#ffffffff",
    degree: tile?.degree ?? 0
  }
}

//---ANIMATION FUNCTIONS---------------------------------------------------------------------
interface AnimationProps {
  xPath: (string | number)[], 
  yPath: (string | number)[], 
  rotatePath: number[], 
  times: number[], 
  totalDuration: number
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
function getAnimationProps (path: number[]) {
  const motionPath = getTileMotionPath(path); // example tile IDs
  const xPath = motionPath.map(p => p.x);
  const yPath = motionPath.map(p => p.y);
  const rotatePath = motionPath.map(p => p.rotate);
  
  const stepCount = motionPath.length;
  const times = Array.from({ length: stepCount }, (_, i) => i / (stepCount - 1));
  const durationPerStep = 0.3;
  const totalDuration = durationPerStep * (stepCount - 1);
  return {xPath: xPath, yPath: yPath, rotatePath: rotatePath, times: times, totalDuration: totalDuration}
}

// ---PAWN VARS-----------------------------------------------------------------------------------------------------------
const pawnColors = ['Red', 'Yellow', 'Green', 'Blue'];
const pawnKeys = ['Pawn1', 'Pawn2', 'Pawn3', 'Pawn4'];
interface Pawn {
  key: string;
  currentTileId: number;
  x: number;
  y: number;
  degree: number;
  color: string;
}

//---IMAGE FUNCTIONS-------------------------------------------------------------------
function makeAnimatedPawn(path: AnimationProps, pawn: Pawn, onClick?: () => void){
  
  // get plane img
  let src = undefined;
  if (pawn.color == tileColors[0]) { //Red
    src = planeRed?.src; 
  }else if (pawn.color == tileColors[1]) {
    src = planeGreen?.src;
  }else if (pawn.color == tileColors[2]) {
    src = planeBlue?.src;
  }else {
    src = planeYellow?.src;
  }

  // generate motion img animation
  return(
     <motion.img
      key={pawn.key}
      src={src}
      alt={pawn.key}
      className={styles.planeImage}
      initial={{
        left: path.xPath[0],
        top: path.yPath[0],
        rotate: path.rotatePath[0]
      }}
      animate={{
        left: path.xPath,
        top: path.yPath,
        rotate: path.rotatePath
      }}
      transition={{
        duration: path.totalDuration,
        ease: "easeInOut",
        times: path.times
      }}      
      onClick={onClick}
      style={{
        position: "absolute",
        width: `6.8%`,
        height: `6.8%`,
        borderRadius: 4,
        transformOrigin: "50% 50%"
      }}
    />
  );
}

function Board() {
  
  // Component Instantiations
  const [tiles, setTiles] = useState<{ id: number; x: number; y: number; size: number; color: string, degree: number }[]>(baseTilesInit);
  const [serverState, setServerState] = useState<any | null>(null);
  const [lastMoveInfo, setLastMoveInfo] = useState<any | null>("No Last Move");
  const [availableMoves, setAvailableMoves] = useState<any | null>(null);
  const [dice, setDice] = useState<number | null>(0)
  const [gameStatus, setGameStatus] = useState(false);
  const [update, setUpdate] = useState<boolean>(false);

  //---Pawn Components Instantiations and Functions-------------------------------------------
  const [currPath, setCurrPath] = useState<number[]>([]);
  const [pawns, setPawns] = useState<Record<string, { pawn: Pawn; path: AnimationProps }>> (
    () => {
      const initial: Record<string, { pawn: Pawn; path: AnimationProps }> = {};
      for (const color of pawnColors) {
        for (let i = 0; i < 4; i++) {
          const key = `${color.toLowerCase()}${pawnKeys[i]}`;
          initial[key] = {
            pawn: getTileById(key, baseTileIds[color][i]),
            path: getAnimationProps([baseTileIds[color][i]])
          };
        }
      }
      return initial;
  });

  function sendPawnBack(key: string, baseTileId: number) {
    setPawns(prev => ({
      ...prev,
      [key]: {
        pawn: {
          ...prev[key].pawn,
          currentTileId: baseTileId
        },
        path: getAnimationProps([prev[key].pawn.currentTileId, baseTileId])
      }
    }));
  }

  function handlePawnClick(key: string, pawn:Pawn, path: number[]) {
    console.log(`Clicked ${key}`);
    // You can update tile or path here
    
    const fullPath = [...(Array.isArray(path) ? path : [])];      
    setPawns(prev => ({
      ...prev,
      [key]: {
        pawn: pawn,
        path: getAnimationProps(fullPath)
      }
    }));
  }

  //---State Functions-----------------------------------------------
  const compareStates = (prevState : any | null, currentState : any | null) => {
    let changes = [];
    
    // Compare players' pawns
    for (let i = 0; i < prevState.players.length; i++) {
      const prevPlayer = prevState.players[i];
      const currentPlayer = currentState.players[i];
      
      for (let j = 0; j < prevPlayer.pawns.length; j++) {
        const prevPawn = prevPlayer.pawns[j];
        const currentPawn = currentPlayer.pawns[j];

        // Check if the tile position has changed
        if (prevPawn.tile_position !== currentPawn.tile_position) {
          changes.push({
            pawn_id: prevPawn.id,
            player_color: prevPlayer.color,
            previous_position: prevPawn.tile_position,
            current_position: currentPawn.tile_position
          });
        }
      }
    }
    
    return changes;
  };
  const findPawnBoardIndex = (state : GameState, pawn_id : string) => {
    const color = pawn_id.split('_')[0];
    const player = state.players.find(player => player.color === color);

    if (player) {
      const pawn = player.pawns.find(pawn => pawn.id === pawn_id);
      if (pawn) {
        return pawn.board_idx;
      }
    }
    return null;
  }
  const getTarget = (stateIndex: number | null, color: string) => {
    let preTarget = 0;  // Initialize preTarget

    if (stateIndex != null) {
      if (stateIndex === -1) {
        // If preStateIndex is -1, determine preTarget based on color
        if (color === 'Red') {
          preTarget = 99;
        } else if (color === 'Yellow') {
          preTarget = 199;
        } else if (color === 'Blue') {
          preTarget = 299;
        } else if (color === 'Green') {
          preTarget = 399;
        }
      } else if (stateIndex >= 50) {
        // If preStateIndex is greater than or equal to 50, reduce preStateIndex by 50
        preTarget = stateIndex - 50;
      }
    } else {
      preTarget = 0;
    }

    return preTarget;
  };

  //---Refresh Board-----------------------------------------------
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://192.168.72:8080/api/play/data");
        const data = await res.json();
        if(data.message){
          refreshBoard();
          alert('HI');
        }

      } catch (err) {
        console.error("Log fetch failed", err);
      }
    }, 1000); // fetch every 1s
    return () => clearInterval(interval);
  }, []);
  


   const refreshBoard = async () => {
    try {
      const gid = localStorage.getItem('game_id');
      if (!gid) return alert('No game in progress');

      const res = await fetch(`http://192.168.72:8080/api/play/refresh`);
      const data = await res.json();

      const prevState = serverState;
      const currentState = data.state;

      const updates = compareStates(prevState, currentState);
     
      updates.forEach(update => {
        const color = update.pawn_id.split('_')[0]
        const idx = update.pawn_id.split('_')[1]
        const pawn_id = `${color.toLowerCase()}Pawn${idx}`

        const path = [];
        let preStateIndex = findPawnBoardIndex(prevState, update.pawn_id);
        let currStateIndex = findPawnBoardIndex(currentState, update.pawn_id);
        
        const preTarget = getTarget(preStateIndex, color);
        const currTarget = getTarget(currStateIndex, color);

        for (let i = preTarget; i <= currTarget; i++) {
          path.push(i);
        }

        pawns[pawn_id].pawn.currentTileId = currTarget;        
        handlePawnClick(pawn_id, pawns[pawn_id].pawn, path);
      });
      
      setServerState(data.state);
    } catch (err) {
 //     alert('refreshBoard error: ' + (err instanceof Error ? err.message : String(err)));
    }
  };

  //---AUDIO SETUP-------------------------------------------------------------------
  const moveAudioRef = useRef<HTMLAudioElement | null>(null);
  const lastMoveRef = useRef<any>(null);

  useEffect(() => {
    try {
      moveAudioRef.current = new Audio('/move-self.mp3');
      moveAudioRef.current.volume = 0.9;
    } catch (e) {
      console.warn('Audio init failed', e);
    }
  }, []);

  //---Board HTML-------------------------------------------------------------------------
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
          {Object.entries(pawns).map(([key, { pawn, path }]) =>
            makeAnimatedPawn(path, pawn, () => handlePawnClick(key, pawn, currPath))
          )}

      </div>
      
      <div>{dice}</div>

      { /*start game button and reset board pawns*/ }
      <button onClick={async ()=> {
        for (const color of pawnColors) {
          for (let i = 0; i < 4; i++) {
            const key = `${color.toLowerCase()}${pawnKeys[i]}`;
            sendPawnBack(key, baseTileIds[color][i]);
          }
        }

        try {
          const res = await fetch('http://192.168.72:8080/api/play/start', { method: 'POST' });
          const data = await res.json();
          const gid = data.game_id;
          // store game id in localStorage for demo; in a real app use context or state
          localStorage.setItem('game_id', gid);
          // map server pawn positions to on-screen pawns
          const state = data.state;
          setServerState(state);
          setDice(data.dice ?? null);
          setGameStatus(true);
        }catch (e) {
          alert("Start Game Button : " + e);
        }
        }}> 
        Start Game
      </button>

      <button onClick={refreshBoard}>Refresh Board</button>

      <button onClick={() => handlePawnClick('redPawn1', pawns['redPawn1'].pawn, [0,1,2])}>ss</button>
  </div>
  );


}

export default Board;