import React, { useState, useEffect, useCallback, useRef} from 'react';
import './App.css';
import io from 'socket.io-client';
import DirectionButtons from './DirectionButtons';

function App() {
  const [isVideoOn, setIsVideoOn] = useState(false);
  const [isDataSending, setIsDataSending] = useState(false);
  const [boxContent, setBoxContent] = useState(""); // Removed initial content
  const [logs, setLogs] = useState([]); // To store logs from Flask
  const [isVideoActive, setIsVideoActive] = useState(false);  // Track if video is active for button color
  const [isDataActive, setIsDataActive] = useState(false); // Track if data transmission is active for button color
  const flaskServerUrl = "http://127.0.0.1:8000";
  const audioContextRef = useRef(null);
  const socket = io('http://127.0.0.1:8000/');
  const [audioBuffer, setAudioBuffer] = useState([]);
  const [imageSrc, setImageSrc] = useState(null);

  useEffect(() => {
    // Initialize WebSocket connection to Flask
    const socket = io(flaskServerUrl, { transports: ['websocket'] });

    socket.on("connect", () => {
      console.log("Connected to Flask WebSocket");
    });

    // Listen for logs from Flask
    socket.on("log", (data) => {
      console.log("Received log:", data.message); // Log received message from Flask
      setLogs(prevLogs => [...prevLogs, data.message]); // Add new log to logs state
    });

    socket.on("disconnect", () => {
      console.log("Disconnected from Flask WebSocket");
    });

    return () => socket.disconnect(); // Cleanup WebSocket on component unmount
  }, []);

// old video feed
//   const toggleVideoFeed = useCallback(() => {
// //     setIsVideoOn(prevState => {
// //       const newState = !prevState;
// //       setIsVideoActive(newState);  // Update button color when video feed is toggled
// //       return newState;
// //     });
// //   }, []);
const toggleVideoFeed = useCallback(async () => {
    const newState = !isVideoOn;
    setIsVideoOn(newState);

    try {
      const action = newState ? "start" : "stop"; // Start or stop the stream
      const response = await fetch(`${flaskServerUrl}/toggle_stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });

      if (!response.ok) {
        throw new Error("Failed to toggle video stream");
      }

      const data = await response.json();
      console.log("Stream status:", data.message);
    } catch (error) {
      console.error("Error toggling video stream:", error);
    }
  }, [isVideoOn, flaskServerUrl]);

  const toggleDataTransmission = async () => {
    const newAction = isDataSending ? 'stop' : 'start';
    setIsDataSending(current => !current); // Toggle UI instantly
    setIsDataActive(current => !current);  // Update button color when data transmission is toggled

    try {
      const response = await fetch(`${flaskServerUrl}/save_data`, {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify({ action: newAction })
      });

      const data = await response.json();
      if (response.ok) {
          setBoxContent(`${newAction} data transmission`);
      } else {
          setIsDataSending(current => !current);
          setBoxContent("Failed to update data transmission");
          throw new Error(data.message || 'Unknown error');
      }
    } catch (error) {
      console.error('Error toggling data transmission:', error);
    }
  };

    console.log('hello');
    socket.on('audio_data', (data) => {
        console.log('Received audio data:', data);
        setAudioBuffer((prevBuffer) => [...prevBuffer, ...data]);
    });
    socket.on("video_frame", (data) => {
        const imageBlob = new Blob([new Uint8Array(data.frame)], { type: "image/jpeg" });
        const imageUrl = URL.createObjectURL(imageBlob);
        setImageSrc(imageUrl);
    });



     const playTestTone = () => {
      const oscillator = audioContextRef.current.createOscillator();
      oscillator.type = 'sine';
      oscillator.frequency.setValueAtTime(440, audioContextRef.current.currentTime); // A4 note
      oscillator.connect(audioContextRef.current.destination);
      oscillator.start();
      oscillator.stop(audioContextRef.current.currentTime + 1); // Play for 1 second
  };

  return (
    <div className="App">
        <header className="App-header">
            {/*<h2>Live Video Feed</h2>*/}



            {/*{isVideoOn && (*/}
            {/*  <div className="video-feed">*/}
            {/*    <img*/}
            {/*      src={`${flaskServerUrl}/stream`}*/}
            {/*      alt="Live Video Feed"*/}
            {/*      onError={(e) => console.error("Error loading video feed:", e)}*/}
            {/*      onLoad={() => console.log("Video feed loaded successfully")}*/}
            {/*      crossOrigin="anonymous"*/}
            {/*      style={{ width: '80%', border: '2px solid #333' }}*/}
            {/*    />*/}
            {/*  </div>*/}
            <div>
                <h1>Real-Time Video Stream</h1>
                {imageSrc && <img src={imageSrc} alt="Video Stream" style={{width: "600px"}}/>}
            </div>
            {/* Video toggle button */}
            <button
                className={`toggle-button ${isVideoActive ? 'active' : ''}`}
                onClick={toggleVideoFeed}
            >
                {isVideoOn ? 'Turn Video Off' : 'Turn Video On'}
            </button>
            <DirectionButtons/>

            {/* Data transmission button */}
            <button
                className={`toggle-button ${isDataActive ? 'active' : ''}`}
                onClick={toggleDataTransmission}
            >
                {isDataSending ? 'Stop Sending Data' : 'Send Data'}
            </button>

            {/* Display Boxes */}
            <div className="info-box">{boxContent}</div>

            {/* Flask log box in the top-right corner */}
            <div className="flask-log-box">
                <h3 className="log-title">Data (not implemented currently)</h3> {/* Title for the log box */}
                {logs.map((log, index) => (
                    <p key={index}>{log}</p>
                ))}
            </div>

            socket.off('audio_data');
            socket.off("video_frame");

        </header>
    </div>
  );
}

export default App;
