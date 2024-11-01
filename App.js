import React, { useEffect, useState } from 'react';

function App() {
    const [data, setData] = useState(null);

    useEffect(() => {
        const fetchData = () => {
            fetch('http://10.138.105.221:5000/api/message') // Flask server endpoint
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => setData(data.message))
                .catch(error => console.error("Error fetching data:", error));
        };

        fetchData(); // Initial fetch
        const intervalId = setInterval(fetchData, 1000); // Fetch every 5 seconds

        return () => clearInterval(intervalId); // Cleanup the interval on component unmount
    }, []);

    return (
        <div>
            <h1>React and Flask Integration!</h1>
            <p>{data ? data : "Loading..."}</p>
        </div>
    );
}

export default App;
