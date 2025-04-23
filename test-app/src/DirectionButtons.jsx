// DirectionButtons.jsx
import React, { useEffect, useState, useRef } from 'react';
import io from 'socket.io-client';
import './index.css';

function DirectionButtons({ logs, setLogs }) {
  /* ───────────────────── State / Refs ───────────────────── */
  const [activeDirections, setActiveDirections] = useState(new Set());
  const [inputMethod, setInputMethod] = useState('buttons');          // 'buttons', 'wasd', or 'gamepad'
  const [previouslyActiveDirections, setPreviouslyActiveDirections] = useState(new Set());
  const [videoPlaying, setVideoPlaying] = useState(false);

  const socketRef = useRef(null);          // <── socket lives here
  const lastActionRef = useRef({});
  const activeButtonsRef = useRef(new Set());

  /* ───────────────────── Socket init ───────────────────── */
  useEffect(() => {
    socketRef.current = io('http://127.0.0.1:8000', { transports: ['websocket'] });

    socketRef.current.on('connect', () =>
      console.log('✅ Web‑Socket connected:', socketRef.current.id)
    );

    socketRef.current.on('disconnect', reason =>
      console.log('⚠️ socket disconnected:', reason)
    );

    return () => socketRef.current?.disconnect();
  }, []);

  /* ───────────────────── Helpers ───────────────────── */
  const sendMovementCommand = (direction, state) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit('movement_command', { direction, state });   // ← now over socket
    } else {
      console.warn('Socket not connected – command dropped');
    }
  };

  const toggleVideo = () => {
    setVideoPlaying(prev => {
      const newState = !prev;
      setLogs(prevLogs => [
        ...prevLogs,
        newState ? 'Video started' : 'Video paused'
      ]);

      if (socketRef.current?.connected) {
        socketRef.current.emit('toggle_video', {action: newState ? 'start' : 'stop'  });
      }

      return newState;
    });
  };

  /* ───────────────────── Game‑pad polling ───────────────────── */
  useEffect(() => {
    const handleGamepadInput = () => {
      const gp = navigator.getGamepads()[0];
      if (!gp) return;

      const dpad = {
        up: gp.buttons[12]?.pressed,
        down: gp.buttons[13]?.pressed,
        left: gp.buttons[14]?.pressed,
        right: gp.buttons[15]?.pressed,
        aButton: gp.buttons[0]?.pressed
      };

      const newActive = new Set();

      for (const [direction, pressed] of Object.entries(dpad)) {
        if (pressed) {
          newActive.add(direction);

          if (!activeButtonsRef.current.has(direction)) {
            if (direction === 'aButton') {
              toggleVideo();
            } else {
              sendMovementCommand(direction, 'move');
            }
            lastActionRef.current[direction] = 'move';
          }
        } else if (activeButtonsRef.current.has(direction) && lastActionRef.current[direction] !== 'stop') {
          sendMovementCommand(direction, 'stop');
          lastActionRef.current[direction] = 'stop';
        }
      }

      activeButtonsRef.current = newActive;
      setActiveDirections(newActive);
    };

    const id = setInterval(handleGamepadInput, 100);
    return () => clearInterval(id);
  }, [logs]);

  /* ───────────────────── WASD handlers ───────────────────── */
  const handleKeyDown = e => {
    if (inputMethod !== 'wasd') return;

    const map = { w: 'up', a: 'left', s: 'down', d: 'right' };
    const dir = map[e.key.toLowerCase()];
    if (!dir) return;

    if (!activeDirections.has(dir)) {
      setActiveDirections(p => new Set(p).add(dir));
      sendMovementCommand(dir, 'move');
      setLogs(l => [...l, `WASD Pressed: ${dir}`]);
    }
  };

  const handleKeyUp = e => {
    if (inputMethod !== 'wasd') return;

    const map = { w: 'up', a: 'left', s: 'down', d: 'right' };
    const dir = map[e.key.toLowerCase()];
    if (!dir) return;

    if (activeDirections.has(dir)) {
      setActiveDirections(p => {
        const n = new Set(p);
        n.delete(dir);
        return n;
      });
      sendMovementCommand(dir, 'stop');
      setLogs(l => [...l, `Stopped moving ${dir}`]);
    }
  };

  useEffect(() => {
    if (inputMethod === 'wasd') {
      window.addEventListener('keydown', handleKeyDown);
      window.addEventListener('keyup', handleKeyUp);
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [inputMethod, activeDirections]);

  /* ───────────────────── UI helpers ───────────────────── */
  const toggleInputMethod = m => {
    setInputMethod(m);
    setActiveDirections(new Set());
    setPreviouslyActiveDirections(new Set());
  };

  /* ───────────────────── JSX ───────────────────── */
  return (
    <div>
      {/* Input‑method selector */}
      <div className="input-method-controls">
        <button
          className={`control-button ${inputMethod === 'wasd' ? 'active' : ''}`}
          onClick={() => toggleInputMethod('wasd')}
        >
          WASD
        </button>
        <button
          className={`control-button ${inputMethod === 'gamepad' ? 'active' : ''}`}
          onClick={() => toggleInputMethod('gamepad')}
        >
          Gamepad
        </button>
      </div>

      {/* Direction HUD */}
      <div className="keyboard">
        <button className={`key ${activeDirections.has('up') ? 'active' : ''}`}>Forward</button>
        <button className={`key ${activeDirections.has('left') ? 'active' : ''}`}>Left</button>
        <button className={`key ${activeDirections.has('down') ? 'active' : ''}`}>Backward</button>
        <button className={`key ${activeDirections.has('right') ? 'active' : ''}`}>Right</button>
      </div>
    </div>
  );
}

export default DirectionButtons;
