import React, { useState, useEffect, useCallback, useRef } from 'react';
import io from 'socket.io-client';
import './App.css';
import DirectionButtons from './DirectionButtons';

const flaskServerUrl = "http://192.168.34.26:8000";
const FRAME_DURATION = 0.1; // seconds per audio chunk (100ms)
const SAMPLE_RATE = 16000; // fixed sample rate for playback
const FRAME_BYTES = SAMPLE_RATE * FRAME_DURATION * Float32Array.BYTES_PER_ELEMENT; // 1600 samples * 4 bytes = 6400 bytes

function App() {
    const [isVideoOn, setIsVideoOn] = useState(false);
    const [boxContent, setBoxContent] = useState("");
    const [logs, setLogs] = useState([]);
    const [imageSrc, setImageSrc] = useState(null);
    const [loading, setLoading] = useState(false);
    const [gamepadState, setGamepadState] = useState({
        up: false,
        down: false,
        left: false,
        right: false,
        aButton: false
    });
    const [audioLabel, setAudioLabel] = useState("Waiting…");

    const socketRef = useRef(null);
    const audioContextRef = useRef(null);
    const nextPlayTimeRef = useRef(0);
    const pending = useRef(new Uint8Array(0));
    const [socketReady, setSocketReady] = useState(false);

    const playAudioChunk = (chunk) => {
        if (!chunk) return;
        const audioContext = audioContextRef.current;
        if (audioContext.state === 'suspended') {
            audioContext.resume();
        }
        const floatData = new Float32Array(chunk);
        console.log(`playAudioChunk: ${floatData.length} samples @ ${SAMPLE_RATE} Hz`);

        const audioBuffer = audioContext.createBuffer(1, floatData.length, SAMPLE_RATE);
        audioBuffer.copyToChannel(floatData, 0, 0);

        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);

        const now = audioContext.currentTime;
        if (nextPlayTimeRef.current < now) {
            nextPlayTimeRef.current = now;
        }
        source.start(nextPlayTimeRef.current);
        nextPlayTimeRef.current += audioBuffer.duration;
    };

    // Initialize AudioContext
    useEffect(() => {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }, []);

    useEffect(() => {
        const socket = io(flaskServerUrl, { transports: ['websocket'] });
        socket.binaryType = 'arraybuffer';
        socketRef.current = socket;

        socket.on("connect", () => { console.log("Connected to Flask WebSocket"); setSocketReady(true)});

        socket.on("log", ({ message: data }) => {
            console.log("Received log:", data);
            setLogs(prev => [...prev, data]);
        });

        /*
        socket.on('audio_data', (data) => {
            const incoming = new Uint8Array(data.chunk);
            const byteCount = incoming.byteLength;
            const sampleCount = byteCount / Float32Array.BYTES_PER_ELEMENT;
            console.log(`Received audio_data: ${byteCount} bytes → ${sampleCount} samples`);
            console.log(`Calculated frame duration: ${(sampleCount / SAMPLE_RATE).toFixed(3)}s`);

            const combined = new Uint8Array(pending.current.length + incoming.length);
            combined.set(pending.current, 0);
            combined.set(incoming, pending.current.length);

            let offset = 0;
            while (combined.length - offset >= FRAME_BYTES) {
                const frame = combined.slice(offset, offset + FRAME_BYTES);
                playAudioChunk(frame.buffer);
                offset += FRAME_BYTES;
            }

            pending.current = combined.slice(offset);
        });*/

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

        socket.on("audio_classification", ({ result }) => {
            console.log("audio classified:", result);
            setAudioLabel(result ? "screaming" : "calm");
        });
        
        return () => {
            audioContextRef.current?.close();
            socket.off("connect");
            socket.off("log");
            socket.off("audio_data");
            socket.off("video_frame");
            socket.off("disconnect");
            socket.off("video_stream_stopped");
            socket.off("audio_classification");
            socket.disconnect();
        };
    }, []);

    const toggleVideoFeed = useCallback(() => {
        audioContextRef.current?.resume();
        setLoading(true);
        const newState = !isVideoOn;
        socketRef.current.emit("toggle_video", { action: newState ? "start" : "stop" });
        socketRef.current.once("video_status", (data) => {
            console.log("Received video status:", data.status);
            setIsVideoOn(data.status === "running");
            if (data.status === "stopped") {
                setImageSrc(null);
            }
            setLoading(false);
        });
    }, [isVideoOn]);

    const updateGamepadState = useCallback(() => {
        const gamepads = navigator.getGamepads();
        if (gamepads && gamepads[0]) {
            const gamepad = gamepads[0];
            const newState = {
                up: gamepad.buttons[12]?.pressed || false,
                down: gamepad.buttons[13]?.pressed || false,
                left: gamepad.buttons[14]?.pressed || false,
                right: gamepad.buttons[15]?.pressed || false,
                aButton: gamepad.buttons[0]?.pressed || false
            };
            setGamepadState(prevState => {
                if (!prevState.aButton && newState.aButton) {
                    toggleVideoFeed();
                }
                return newState;
            });
        }
    }, [toggleVideoFeed]);

    useEffect(() => {
        const interval = setInterval(updateGamepadState, 100);
        return () => clearInterval(interval);
    }, [updateGamepadState]);

    return (
        <div className="App">
            <header className="App-header">
                <h1 style={{ color: '#93cf40' }}>SET RescueGATOR</h1>
                {imageSrc && <img src={imageSrc} alt="Video Stream" style={{ width: "750px", border: "5px solid #0f6cb6"}} />}
                <button
                    className={`toggle-button ${isVideoOn ? 'active' : ''}`}
                    onClick={toggleVideoFeed}
                    disabled={loading}
                >{loading ? "Processing..." : isVideoOn ? "Turn Video Off" : "Turn Video On"}</button>
                {socketReady && <DirectionButtons socket={socketRef.current} gamepadState={gamepadState} logs={logs} setLogs={setLogs}/>}
                <div className="info-box">{boxContent}</div>
                <div className="flask-log-box">
                  <h3 className="log-title">Logs</h3>
                  {[...logs].reverse().map((log, idx) => (
                     <p key={idx}>{log}</p>
                  ))}
                </div>
            </header>
                {/* AUDIO STATUS BOX */}
            <div className="audio-status-box">
                <span
                    className={
                        audioLabel === "screaming" ? "label-scream" : "label-calm"
                    }
                >
                    {audioLabel}
                </span>
            </div>
            <div className="gamepad-legend">
                <h2>Controller Legend</h2>
                <ul>
                    <li><strong>D-Pad Up:</strong> Forward</li>
                    <li><strong>D-Pad Down:</strong> Back</li>
                    <li><strong>D-Pad Left:</strong> Left</li>
                    <li><strong>D-Pad Right:</strong> Right</li>
                    <li><strong>A Button:</strong> Toggle Video</li>
                </ul>
            </div>
        </div>
    );
}

export default App;
