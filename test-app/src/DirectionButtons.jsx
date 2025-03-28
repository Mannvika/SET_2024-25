import React, { useEffect, useState, useRef } from 'react';
import './index.css';

function DirectionButtons({ logs, setLogs }) {
    const [activeDirections, setActiveDirections] = useState(new Set());
    const [inputMethod, setInputMethod] = useState('buttons'); // 'buttons', 'wasd', or 'gamepad'
    const [previouslyActiveDirections, setPreviouslyActiveDirections] = useState(new Set()); // Track previous active directions
    const [videoPlaying, setVideoPlaying] = useState(false); // Track the state of the video (playing or paused)
    const lastActionRef = useRef({});
    const activeButtonsRef = useRef(new Set());

    // Function to toggle video
    const toggleVideo = () => {
        setVideoPlaying(prevState => {
            const newState = !prevState;
            setLogs(prevLogs => [
                ...prevLogs,
                newState ? "Video started" : "Video paused"
            ]);
            return newState;
        });
    };

    // Send command to Flask server
    const sendCommand = async (direction, state) => {
        try {
            const response = await fetch('http://127.0.0.1:8000/direction', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ direction, state }),
            });
            const data = await response.json();
            console.log("Server response:", data);
        } catch (error) {
            console.error("Error sending command:", error);
        }
    };

    useEffect(() => {
        const handleGamepadInput = () => {
            const gamepads = navigator.getGamepads();
            if (gamepads[0]) {
                const gamepad = gamepads[0];

                // Map buttons
                const dpadButtons = {
                    up: gamepad.buttons[12]?.pressed,
                    down: gamepad.buttons[13]?.pressed,
                    left: gamepad.buttons[14]?.pressed,
                    right: gamepad.buttons[15]?.pressed,
                    aButton: gamepad.buttons[0]?.pressed, // A button is typically at index 0
                };

                const newActiveButtons = new Set();

                for (const [direction, isPressed] of Object.entries(dpadButtons)) {
                    if (isPressed) {
                        newActiveButtons.add(direction);

                        // Only log and send if it's a new press
                        if (!activeButtonsRef.current.has(direction)) {
                            setLogs(prevLogs => [...prevLogs, `Button Pressed: ${direction}`]);
                            if (direction === 'aButton') {
                                toggleVideo(); // Trigger the toggle video function when A button is pressed
                            } else {
                                sendCommand(direction, 'move');
                            }
                            lastActionRef.current[direction] = "move"; // Update last action
                        }
                    } else if (activeButtonsRef.current.has(direction)) {
                        if (lastActionRef.current[direction] !== "stop") {
                            setLogs(prevLogs => [...prevLogs, `Button for ${direction} stopped being pressed`]);
                            sendCommand(direction, 'stop');
                            lastActionRef.current[direction] = "stop";
                        }
                    }
                }

                activeButtonsRef.current = newActiveButtons;
                setActiveDirections(newActiveButtons); // Update the active directions state
            }
        };

        const gamepadInterval = setInterval(handleGamepadInput, 100);
        return () => clearInterval(gamepadInterval);
    }, [logs]);

    // Handle WASD Input
    const handleKeyDown = (event) => {
        if (inputMethod !== 'wasd') return;

        let direction = null;
        switch (event.key) {
            case 'w':
            case 'W':
                direction = 'up';
                break;
            case 'a':
            case 'A':
                direction = 'left';
                break;
            case 's':
            case 'S':
                direction = 'down';
                break;
            case 'd':
            case 'D':
                direction = 'right';
                break;
            default:
                return;
        }

        if (!activeDirections.has(direction)) {
            setActiveDirections(prev => new Set(prev.add(direction)));
            sendCommand(direction, 'move');
            setLogs(prevLogs => [...prevLogs, `WASD Pressed: ${direction}`]);
        }
    };

    const handleKeyUp = (event) => {
        if (inputMethod !== 'wasd') return;

        let direction = null;
        switch (event.key) {
            case 'w':
            case 'W':
                direction = 'up';
                break;
            case 'a':
            case 'A':
                direction = 'left';
                break;
            case 's':
            case 'S':
                direction = 'down';
                break;
            case 'd':
            case 'D':
                direction = 'right';
                break;
            default:
                return;
        }

        if (activeDirections.has(direction)) {
            setActiveDirections(prev => {
                const newDirections = new Set(prev);
                newDirections.delete(direction);
                return newDirections;
            });
            sendCommand(direction, 'stop');
            setLogs(prevLogs => [...prevLogs, `Stopped moving ${direction}`]);
        }
    };

    useEffect(() => {
        if (inputMethod === 'wasd') {
            window.addEventListener('keydown', handleKeyDown);
            window.addEventListener('keyup', handleKeyUp);
        } else {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
        }

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
        };
    }, [inputMethod, activeDirections]);

    const toggleInputMethod = (method) => {
        setInputMethod(method);
        setActiveDirections(new Set());
        setPreviouslyActiveDirections(new Set());
    };

    return (
        <div>
            {/* Toggle buttons for input method */}
            <div className="input-method-controls">
                <button
                    className={`control-button ${inputMethod === 'buttons' ? 'active' : ''}`}
                    onClick={() => toggleInputMethod('buttons')}
                >
                    Button Controls
                </button>
                <button
                    className={`control-button ${inputMethod === 'wasd' ? 'active' : ''}`}
                    onClick={() => toggleInputMethod('wasd')}
                >
                    WASD Controls
                </button>
                <button
                    className={`control-button ${inputMethod === 'gamepad' ? 'active' : ''}`}
                    onClick={() => toggleInputMethod('gamepad')}
                >
                    Gamepad Controls
                </button>
            </div>

            {/* Directional Buttons */}
            <div className="keyboard">
                <button className={`key ${activeDirections.has('up') ? 'active' : ''}`}>Up</button>
                <button className={`key ${activeDirections.has('left') ? 'active' : ''}`}>Left</button>
                <button className={`key ${activeDirections.has('down') ? 'active' : ''}`}>Down</button>
                <button className={`key ${activeDirections.has('right') ? 'active' : ''}`}>Right</button>
            </div>

        </div>
    );
}

export default DirectionButtons;
