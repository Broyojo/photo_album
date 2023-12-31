import React, { useState } from 'react';

const SearchBar = ({ onSearch }) => {
    const [query, setQuery] = useState("");
    const [startDateTime, setStartDateTime] = useState("");
    const [endDateTime, setEndDateTime] = useState("");
    const [radius, setRadius] = useState(10);
    const [address, setAddress] = useState("");

    const handleSubmit = (event) => {
        event.preventDefault();
        onSearch(query, startDateTime, endDateTime, address, radius);
    };

    return (
        <form onSubmit={handleSubmit}>
            <label htmlFor="search-query">
                Search Query:
                <input
                    id="search-query"
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search for media..."
                />
            </label>
            <label htmlFor="start-date-time">
                Start Date and Time:
                <input
                    id="start-date-time"
                    type="datetime-local"
                    value={startDateTime}
                    onChange={(e) => setStartDateTime(e.target.value)}
                />
            </label>
            <label htmlFor="end-date-time">
                End Date and Time:
                <input
                    id="end-date-time"
                    type="datetime-local"
                    value={endDateTime}
                    onChange={(e) => setEndDateTime(e.target.value)}
                />
            </label>
            <label htmlFor="address">
                Address:
                <input
                    id="address"
                    type="text"
                    value={address}
                    onChange={(e) => setAddress(e.target.value)}
                    placeholder="Enter address"
                />
            </label>
            <label htmlFor="radius">
                Radius (meters):
                <input
                    id="radius"
                    type="number"
                    value={radius}
                    min={1}
                    onChange={(e) => setRadius(e.target.value)}
                    placeholder="Enter radius in meters"
                />
            </label>
            <button type="submit">Search</button>
        </form>
    )
}

export default SearchBar;
