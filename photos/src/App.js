import './App.css';

import axios from 'axios';
import React, { useState } from 'react';
import './App.css';
import MediaGallery from './components/MediaGallery';
import SearchBar from './components/SearchBar';

const App = () => {
    const [allMedia, setAllMedia] = useState([]);

    const handleSearch = async (query, startDateTime, endDateTime, address, radius) => {
        let data = {
            query: query,
            count: 1000,
        };
        if (startDateTime !== "") {
            data.startDateTime = startDateTime;
        }
        if (endDateTime !== "") {
            data.endDateTime = endDateTime;
        }
        if (address !== "") {
            data.address = address;
        }
        if (radius !== "") {
            data.radius = radius;
        }
        console.log(data);
        let response = await axios.post("http://192.168.1.69:8000/api/search",).catch(alert);
        console.log(response);
        setAllMedia(response.data);
    };

    return (
        <div className="App">
            <SearchBar onSearch={handleSearch} />
            <MediaGallery numPerPage={10} allMedia={allMedia} />
        </div>
    );
};

export default App;
