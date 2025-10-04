import { useEffect, useState } from 'react';

import Board from '@/components/Board';
import Test from '@/components/test';

import dice1 from '../../img/1.png';
import dice2 from '../../img/2.png';
import dice3 from '../../img/3.png';
import dice4 from '../../img/4.png';
import dice5 from '../../img/5.png';
import dice6 from '../../img/6.png';
import diceGif from '../../img/dice-game.gif';

interface Turn {
  current_player: string,
  available: boolean,
  dice: number
}

function getDiceSrc(dice: number | null | undefined) {
  const d = Number(dice) || 0;
  switch (d) {
    case 1:
      return (dice1 as any)?.src || dice1;
    case 2:
      return (dice2 as any)?.src || dice2;
    case 3:
      return (dice3 as any)?.src || dice3;
    case 4:
      return (dice4 as any)?.src || dice4;
    case 5:
      return (dice5 as any)?.src || dice5;
    case 6:
      return (dice6 as any)?.src || dice6;
    default:
      return undefined;
  }
}

// map player names to colors (keep in sync with Board.tsx mapping)
const playerColorMap: Record<string, string> = {
  Red: '#ef4444',
  Blue: '#3b82f6',
  Green: '#10b981',
  Yellow: '#fbbf24'
};

function hexToRgba(hex: string, alpha = 1) {
  if (!hex) return `rgba(255,255,255,${alpha})`;
  const h = hex.replace('#', '');
  const bigint = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export default function Home() {

    const [data, setData] = useState("loading...");
    const [turn, setTurn] = useState<Turn | null>(null);
    const [latestLog, setLatestLog] = useState<string>("");
  const [diceAnimating, setDiceAnimating] = useState(false);
  const [rolling, setRolling] = useState(false);
  const [displayedDice, setDisplayedDice] = useState<number | null>(null);

  const [actionCompleted, setActionCompleted] = useState(false);  
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("http://localhost:8080/api/play/data");
        const res2 = await fetch("http://localhost:8080/api/log/latest");
    
        const apiData = await res.json();
        const data = await res2.json();

        setTurn(
          {
          'current_player': apiData.current_player,
          'available': apiData.available,
          'dice': apiData.dice
          }
        );
        setLatestLog(data.latest_log);

      } catch (err) {
        console.error("Log fetch failed", err);
      }
    };

    // Check if the data is available and the action hasn't been completed
    if (data && actionCompleted) {
      fetchData(); // Call the async function here
      setActionCompleted(false); // Mark the action as completed for next round
    }
  }, [data, actionCompleted]);

  // animate dice when value changes
  useEffect(() => {
    // when the server reports a new dice roll, play the rolling GIF first, then show the new face
    if (typeof turn?.dice === 'undefined' || turn?.dice === null) return;

    setRolling(true);
    setDiceAnimating(true);
    const rollDuration = 850; // ms to show GIF
    const t = setTimeout(() => {
      setRolling(false);
      setDisplayedDice(turn?.dice ?? null);
      setDiceAnimating(false);
    }, rollDuration);
    return () => clearTimeout(t);
  }, [turn?.dice]);

   useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:8080/api/data");
        const data = await res.json();
        setData(data.message ? "Updating Turn.." : "Updated Turn...");
        setActionCompleted(data.message); // Reset actionCompleted if needed
      } catch (err) {
        console.error("Log fetch failed", err);
      }
    }, 1000); // fetch every 1s
    return () => clearInterval(interval);
  }, []);


  return (

    <div className="min-h-screen bg-gray-200 flex items-center justify-center p-6">
      <div className="w-full max-w-6xl flex items-center justify-center gap-8">
        {/* Left control panel */}
        <div
          className="w-full max-w-sm mx-auto rounded-lg shadow-md border-2 p-4 font-sans text-center"
          style={{
            background: hexToRgba(playerColorMap[turn?.current_player || ''] || '#e6eef8', 0.12),
            borderColor: playerColorMap[turn?.current_player || ''] || '#3b82f6'
          }}
        >
          <div className="text-lg font-semibold">Player: {turn?.current_player}
          <div className="mt-2">Dice: {turn?.dice} {' '}
            {turn?.dice ? (
              rolling ? (
                <img src={(diceGif as any)?.src || diceGif} alt="dice-rolling" className={diceAnimating ? 'fly_up animate' : 'fly_up'} style={{ width: 48, height: 48, verticalAlign: 'middle', marginLeft: 8 }} />
              ) : (
                <img src={getDiceSrc(displayedDice ?? turn?.dice)} alt={`dice-${displayedDice ?? turn?.dice}`} className={diceAnimating ? 'fly_up animate' : 'fly_up'} style={{ width: 48, height: 48, verticalAlign: 'middle', marginLeft: 8 }} />
              )
            ) : null}
          </div></div>
          <div className="mt-2">Available: {String(turn?.available)}</div>
          <div className="mt-4 text-sm text-gray-700">{latestLog}</div>
        </div>

        {/* Center board area */}
        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="mb-4 text-2xl font-bold">AeroPlane Chess</div>
          <Board />
        </div>
      </div>

      {/* global test.css contains transform classes like .fly_up */}
      {turn?.dice ? (
        rolling ? (
          <img src={(diceGif as any)?.src || diceGif} className={diceAnimating ? 'fly_up animate' : 'fly_up'} alt="dice-rolling" style={{ position: 'fixed', right: 16, bottom: 16, width: 64, height: 64, zIndex: 1200 }} />
        ) : (
          <img src={getDiceSrc(displayedDice ?? turn?.dice)} className={diceAnimating ? 'fly_up animate' : 'fly_up'} alt={`dice-${displayedDice ?? turn?.dice}`} style={{ position: 'fixed', right: 16, bottom: 16, width: 64, height: 64, zIndex: 1200 }} />
        )
      ) : null}

      <Test />
    </div>
  );
}

