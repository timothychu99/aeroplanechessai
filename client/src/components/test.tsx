import React, { useRef } from 'react';
import { motion, useAnimation } from 'framer-motion';
import styles from '@/styles/Test2.module.css';

const TILE_SIZE = 110; // 100px tile + 10px gap

const getCoordinates = (index: number) => {
  const col = index % 3;
  const row = Math.floor(index / 3);
  return { x: col * TILE_SIZE, y: row * TILE_SIZE };
};

const Test = () => {
  const controls = useAnimation(); // motion controls
  const pathRef = useRef<number[]>([]); // for optional debugging

  const handleMove = async (path: number[]) => {
    pathRef.current = path; // optional: for debugging/logging

    for (let i = 0; i < path.length; i++) {
      const coords = getCoordinates(path[i]);
      await controls.start({
        x: coords.x,
        y: coords.y,
        transition: {
          type: 'tween',
          stiffness: 300,
          damping: 25,
        },
      });
    }
  };

  return (
    <div className={styles.wrapper}>
      <div className={styles.grid}>
        {/* Static tiles */}
        {Array.from({ length: 9 }).map((_, index) => (
          <div
            key={index}
            className={styles.tile}
            onClick={() => {
              // Example path: from 0 → index via middle tile 4
              const currentPath = [99, 0, 1, 2, index];
              handleMove(currentPath);
            }}
          >
            {index}
          </div>
        ))}

        {/* Red box that moves through path */}
        <motion.div
          className={styles.redBox}
          animate={controls}
          initial={getCoordinates(0)}
        />
      </div>
    </div>
  );
};

export default Test;
