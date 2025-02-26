import React, { useEffect, useRef, useState} from 'react';
import io from 'socket.io-client'; 

function App() {
    const audioContextRef = useRef(null);
    
    const socket = io('http://127.0.0.1:8000/');
    const [audioBuffer, setAudioBuffer] = useState([]);
    const [imageSrc, setImageSrc] = useState(null);
    useEffect(() => {
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
      

        return () => {
            socket.off('audio_data');
            socket.off("video_frame");
        };
    }, []);

    const playTestTone = () => {
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
      </div>
    );
}

export default App;
