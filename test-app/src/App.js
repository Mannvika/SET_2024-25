import React, { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';

function App() {
    const audioContextRef = useRef(null);
    const [imageSrc, setImageSrc] = useState(null);
    const socketRef = useRef(null);

    useEffect(() => {
        socketRef.current = io("http://localhost:8000", {
            transports: ["websocket"],
            upgrade: false
        });

        // ... existing socket handlers ...

        return () => {
            if (socketRef.current) {
                socketRef.current.disconnect();
            }
        };
    }, []);

    const sendMovementCommand = (action, state) => {
        if (socketRef.current?.connected) {
            socketRef.current.emit('movement_command', {
                action,
                state
            });
        }
    };

    const DirectionalPad = () => (
        <div style={dpadStyle}>
            <button 
                style={buttonStyle}
                onMouseDown={() => sendMovementCommand('forward', true)}
                onMouseUp={() => sendMovementCommand('forward', false)}
                onTouchStart={() => sendMovementCommand('forward', true)}
                onTouchEnd={() => sendMovementCommand('forward', false)}
            >
                ↑
            </button>
            <div style={rowStyle}>
                <button 
                    style={buttonStyle}
                    onMouseDown={() => sendMovementCommand('left', true)}
                    onMouseUp={() => sendMovementCommand('left', false)}
                    onTouchStart={() => sendMovementCommand('left', true)}
                    onTouchEnd={() => sendMovementCommand('left', false)}
                >
                    ←
                </button>
                <button 
                    style={buttonStyle}
                    onMouseDown={() => sendMovementCommand('right', true)}
                    onMouseUp={() => sendMovementCommand('right', false)}
                    onTouchStart={() => sendMovementCommand('right', true)}
                    onTouchEnd={() => sendMovementCommand('right', false)}
                >
                    →
                </button>
            </div>
            <button 
                style={buttonStyle}
                onMouseDown={() => sendMovementCommand('backward', true)}
                onMouseUp={() => sendMovementCommand('backward', false)}
                onTouchStart={() => sendMovementCommand('backward', true)}
                onTouchEnd={() => sendMovementCommand('backward', false)}
            >
                ↓
            </button>
            <button 
                style={{ ...buttonStyle, gridColumn: '1 / span 2' }}
                onMouseDown={() => sendMovementCommand('turn', true)}
                onMouseUp={() => sendMovementCommand('turn', false)}
                onTouchStart={() => sendMovementCommand('turn', true)}
                onTouchEnd={() => sendMovementCommand('turn', false)}
            >
                TURN
            </button>
        </div>
    );

    return (
        <div style={containerStyle}>
            <h1>Real-Time Video Stream</h1>
            {imageSrc && <img src={imageSrc} alt="Video Stream" style={imageStyle} />}
            <DirectionalPad />
        </div>
    );
}

// Style constants
const containerStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '20px'
};

const dpadStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '10px',
    padding: '20px',
    backgroundColor: '#f0f0f0',
    borderRadius: '10px'
};

const buttonStyle = {
    padding: '15px 25px',
    fontSize: '1.2rem',
    cursor: 'pointer',
    border: '2px solid #333',
    borderRadius: '5px',
    transition: 'all 0.2s ease',
    touchAction: 'manipulation'
};

const rowStyle = {
    gridColumn: '1 / span 2',
    display: 'flex',
    justifyContent: 'space-between',
    gap: '10px'
};

const imageStyle = {
    width: '600px',
    borderRadius: '8px',
    boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
};

export default App;
