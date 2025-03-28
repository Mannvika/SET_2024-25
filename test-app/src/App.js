import React, { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';
import io from 'socket.io-client';
import DirectionButtons from './DirectionButtons';

const flaskServerUrl = "http://127.0.0.1:8000";

function App() {
    const [isVideoOn, setIsVideoOn] = useState(false);
    const [isDataSending, setIsDataSending] = useState(false);
    const [boxContent, setBoxContent] = useState("");
    const [logs, setLogs] = useState([]);
    const [imageSrc, setImageSrc] = useState(null);
    const [audioBuffer, setAudioBuffer] = useState([]);
    const audioContextRef = useRef(null);
    const socketRef = useRef(null); // Store WebSocket instance
    const [loading, setLoading] = useState(false);
    const [gamepadState, setGamepadState] = useState({ up: false, down: false, left: false, right: false, aButton: false });

    useEffect(() => {
        // Initialize WebSocket
        socketRef.current = io(flaskServerUrl, { transports: ['websocket'] });

        socketRef.current.on("connect", () => {
            console.log("Connected to Flask WebSocket");
        });

        socketRef.current.on("log", (data) => {
            console.log("Received log:", data.message);
            setLogs(prevLogs => [...prevLogs, data.message]);
        });

        socketRef.current.on("audio_data", (data) => {
            console.log("Received audio data");
            setAudioBuffer(prevBuffer => [...prevBuffer, ...data]);
        });

        socketRef.current.on("video_frame", (data) => {
            const imageBlob = new Blob([new Uint8Array(data.frame)], { type: "image/jpeg" });
            const imageUrl = URL.createObjectURL(imageBlob);
            setImageSrc(imageUrl);
        });

        socketRef.current.on("disconnect", () => {
            console.log("Disconnected from Flask WebSocket");
        });

        socketRef.current.on("video_stream_stopped", () => {
            console.log("Video stream was stopped by the server.");
            setIsVideoOn(false);
        });

        return () => {
            socketRef.current.disconnect(); // Cleanup on unmount
            socketRef.current.off("log");
            socketRef.current.off("audio_data");
            socketRef.current.off("video_frame");
            socketRef.current.off("disconnect");
            socketRef.current.off("video_stream_stopped");
        };
    }, []);

    useEffect(() => {
        if (imageSrc) {
            return () => URL.revokeObjectURL(imageSrc); // Free memory when image updates
        }
    }, [imageSrc]);

    useEffect(() => {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }, []);

    const toggleVideoFeed = useCallback(async () => {
        setLoading(true);  // Show "Processing..." immediately

        const newState = !isVideoOn;

        try {
            socketRef.current.emit("toggle_video", { action: newState ? "start" : "stop" });

            // Listen for confirmation event from the server
            socketRef.current.once("video_status", (data) => {
                console.log("Received video status:", data.status);
                setIsVideoOn(data.status === "running");

                if (data.status === "stopped") {
                    setImageSrc(null); // ✅ Clear the last frame when stopping video
                }

                setLoading(false); // Stop showing "Processing..."
            });

        } catch (error) {
            console.error("Error toggling video stream:", error);
            setLoading(false);
        }
    }, [isVideoOn]);

    const toggleDataTransmission = async () => {
        const newAction = isDataSending ? 'stop' : 'start';
        setIsDataSending(!isDataSending);

        try {
            const response = await fetch(`${flaskServerUrl}/save_data`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: newAction })
            });

            if (response.ok) {
                setBoxContent(`${newAction} data transmission`);
            } else {
                setIsDataSending(prev => !prev);
                setBoxContent("Failed to update data transmission");
                throw new Error("Error updating data transmission");
            }
        } catch (error) {
            console.error('Error toggling data transmission:', error);
        }
    };

    // Detect gamepad state
    const updateGamepadState = () => {
        const gamepads = navigator.getGamepads();
        if (gamepads[0]) {
            const gamepad = gamepads[0];
            const newState = {
                up: gamepad.buttons[12].pressed,
                down: gamepad.buttons[13].pressed,
                left: gamepad.buttons[14].pressed,
                right: gamepad.buttons[15].pressed,
                aButton: gamepad.buttons[0].pressed, // A button is typically at index 0
            };

            setGamepadState(prevState => {
                // Only trigger the toggle action when A button is pressed
                if (!prevState.aButton && newState.aButton) {
                    toggleVideoFeed();  // Toggle video feed when A is pressed
                }
                return newState;
            });
        }
    };

    useEffect(() => {
        const interval = setInterval(updateGamepadState, 100); // Check the gamepad state every 100ms
        return () => clearInterval(interval);
    }, [gamepadState, toggleVideoFeed]);

    return (
        <div className="App">
            <header className="App-header">
                <h1>Video Stream</h1>
                {imageSrc && <img src={imageSrc} alt="Video Stream" style={{ width: "600px" }} />}
                <button
                    className={`toggle-button ${isVideoOn ? 'active' : ''}`}
                    onClick={toggleVideoFeed}
                    disabled={loading} // Disable during operation
                >
                    {loading ? (
                        'Processing...'
                    ) : isVideoOn ? 'Turn Video Off' : 'Turn Video On'}
                </button>

                <DirectionButtons gamepadState={gamepadState} logs={logs} setLogs={setLogs} />

                <button className={`toggle-button ${isDataSending ? 'active' : ''}`} onClick={toggleDataTransmission}>
                    {isDataSending ? 'Stop Sending Data' : 'Send Data'}
                </button>

                <div className="info-box">{boxContent}</div>

                <div className="flask-log-box">
                    <h3 className="log-title">Logs</h3>
                    {logs.map((log, index) => <p key={index}>{log}</p>)}
                </div>
            </header>
        </div>
    );
}

export default App;
