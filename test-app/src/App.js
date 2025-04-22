import React, { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';

function App() {
    const audioContextRef = useRef(null);
    const [audioBuffer, setAudioBuffer] = useState([]);
    const [imageSrc, setImageSrc] = useState(null);

    useEffect(() => {
        const socket = io("http://localhost:8000", {
            transports: ["websocket"],
            upgrade: false
        });

        console.log('Attempting to connect to the socket server...');

        socket.on('connect', () => {
            console.log(`Socket connected: ${socket.id}`);
        });

        socket.on('disconnect', (reason) => {
            console.log(`Socket disconnected: ${reason}`);
        });

        socket.on('connect_error', (error) => {
            console.error('Socket connection error:', error);
        });

        socket.on("video_frame", (data) => {
            console.log('Received video frame');
            console.log('Audio data:', data.audio_data);

            const imageBlob = new Blob([new Uint8Array(data.frame)], { type: "image/h264" });
            const imageUrl = URL.createObjectURL(imageBlob);
            setImageSrc(imageUrl);
        });

        return () => {
            console.log('Cleaning up: Disconnecting socket...');
            socket.off('connect');
            socket.off('disconnect');
            socket.off('connect_error');
            socket.off("video_frame");
            socket.disconnect();
        };
    }, []);

    async function sendArduinoCommand() {
        try {
          // Make a POST request to the backend endpoint with a JSON payload
          const command = "h";
          const response = await fetch('http://localhost:8000/arduino-command', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'  // Inform the server that we are sending JSON
            },
            body: JSON.stringify({ command })     // Convert the command into a JSON string
          });
      
          // Convert the response into JSON
          const data = await response.json();
      
          // Check if the response indicates success
          if (response.ok) {
            console.log('Command sent successfully:', data);
          } else {
            // If there's an error response from the server, log the error message
            console.error('Error sending command:', data.error);
          }
        } catch (error) {
          // Catch and log any network-level errors
          console.error('Network error:', error);
        }
      }

    const playTestTone = () => {
        if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
        }

        const oscillator = audioContextRef.current.createOscillator();
        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(440, audioContextRef.current.currentTime); // A4 note
        oscillator.connect(audioContextRef.current.destination);
        oscillator.start();
        oscillator.stop(audioContextRef.current.currentTime + 1); // Play for 1 second
    };

    return (
        <div>
            <h1>Real-Time Video Stream</h1>
            {imageSrc && <img src={imageSrc} alt="Video Stream" style={{ width: "600px" }} />}
            <button onClick={sendArduinoCommand}>arudino test button</button>
            <button onClick={playTestTone}>Play Test Tone</button>
        </div>
    );
}

export default App;
