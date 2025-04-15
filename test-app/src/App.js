import React, { useState, useEffect, useCallback, useRef } from 'react';
import io from 'socket.io-client';
import './App.css';
import DirectionButtons from './DirectionButtons';

const flaskServerUrl = "http://127.0.0.1:8000";

function App() {
    const [isVideoOn, setIsVideoOn] = useState(false);
    const [boxContent, setBoxContent] = useState("");
    const [logs, setLogs] = useState([]);
    const [imageSrc, setImageSrc] = useState(null);
    const [audioBuffer, setAudioBuffer] = useState([]);
    const [loading, setLoading] = useState(false);
    const [gamepadState, setGamepadState] = useState({
        up: false,
        down: false,
        left: false,
        right: false,
        aButton: false
    });

    const socketRef = useRef(null);
    const audioContextRef = useRef(null);

    // Socket initialization and cleanup
    useEffect(() => {
        const socket = io(flaskServerUrl, { transports: ['websocket'] });
        socketRef.current = socket;

        socket.on("connect", () => console.log("Connected to Flask WebSocket"));

        socket.on("log", (data) => {
            console.log("Received log:", data.message);
            setLogs(prevLogs => [...prevLogs, data.message]);
        });

        socket.on("audio_data", (data) => {
            console.log("Received audio data");
            setAudioBuffer(prevBuffer => [...prevBuffer, ...data]);
        });

        socket.on("video_frame", (data) => {
            const imageBlob = new Blob([new Uint8Array(data.frame)], { type: "image/jpeg" });
            const imageUrl = URL.createObjectURL(imageBlob);
            setImageSrc(imageUrl);
        });

        socket.on("disconnect", () => {
            console.log("Disconnected from Flask WebSocket");
        });

        socket.on("video_stream_stopped", () => {
            console.log("Video stream was stopped by the server.");
            setIsVideoOn(false);
        });

        return () => {
            socket.off("connect");
            socket.off("log");
            socket.off("audio_data");
            socket.off("video_frame");
            socket.off("disconnect");
            socket.off("video_stream_stopped");
            socket.disconnect();
        };
    }, []);

    // Cleanup image URLs to free up memory
    useEffect(() => {
        return () => {
            if (imageSrc) {
                URL.revokeObjectURL(imageSrc);
            }
        };
    }, [imageSrc]);

    // Initialize AudioContext on mount
    useEffect(() => {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }, []);

    // Toggle video feed callback
    const toggleVideoFeed = useCallback(() => {
        setLoading(true);
        const newState = !isVideoOn;

        socketRef.current.emit("toggle_video", { action: newState ? "start" : "stop" });
        socketRef.current.once("video_status", (data) => {
            console.log("Received video status:", data.status);
            setIsVideoOn(data.status === "running");
            if (data.status === "stopped") {
                setImageSrc(null); // Clear the last frame when stopping
            }
            setLoading(false);
        });
    }, [isVideoOn]);

    // Update gamepad state and toggle video on A-button press
    const updateGamepadState = useCallback(() => {
        const gamepads = navigator.getGamepads();
        if (gamepads && gamepads[0]) {
            const gamepad = gamepads[0];
            const newState = {
                up: gamepad.buttons[12]?.pressed || false,
                down: gamepad.buttons[13]?.pressed || false,
                left: gamepad.buttons[14]?.pressed || false,
                right: gamepad.buttons[15]?.pressed || false,
                aButton: gamepad.buttons[0]?.pressed || false,
            };

            setGamepadState(prevState => {
                // When A button is newly pressed, toggle video feed
                if (!prevState.aButton && newState.aButton) {
                    toggleVideoFeed();
                }
                return newState;
            });
        }
    }, [toggleVideoFeed]);

    // Check gamepad state every 100ms
    useEffect(() => {
        const interval = setInterval(updateGamepadState, 100);
        return () => clearInterval(interval);
    }, [updateGamepadState]);

    return (
        <div className="App">
            <header className="App-header">
                <h1 style={{ color: '#93cf40' }}>SET RescueGATOR</h1>
                {imageSrc && <img src={imageSrc} alt="Video Stream" style={{ width: "600px" }} />}
                <button
                    className={`toggle-button ${isVideoOn ? 'active' : ''}`}
                    onClick={toggleVideoFeed}
                    disabled={loading}
                >
                    {loading ? "Processing..." : isVideoOn ? "Turn Video Off" : "Turn Video On"}
                </button>
                <DirectionButtons gamepadState={gamepadState} logs={logs} setLogs={setLogs} />
                <div className="info-box">{boxContent}</div>
                <div className="flask-log-box">
                    <h3 className="log-title">Logs</h3>
                    {logs.map((log, index) => <p key={index}>{log}</p>)}
                </div>
            </header>

            {/* Controller Legend displayed in the bottom right-hand side */}
            <div className="gamepad-legend">
                <h2>Controller Legend</h2>
                <ul>
                    <li><strong>D-Pad Up:</strong> Move Forward</li>
                    <li><strong>D-Pad Down:</strong> Move Back</li>
                    <li><strong>D-Pad Left:</strong> Move Left</li>
                    <li><strong>D-Pad Right:</strong> Move Right</li>
                    <li><strong>A Button:</strong> Toggle Video Feed</li>
                </ul>
            </div>
        </div>
    );
}

export default App;
