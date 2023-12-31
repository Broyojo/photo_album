import React, { useEffect, useState } from 'react';

const Media = ({ media, onClick }) => {
    const [isLoading, setIsLoading] = useState(true);
    const mediaStyle = {
        margin: '5px',
        flex: '0 0 auto',
    };

    const imageStyle = {
        width: '150px',
        height: 'auto',
        display: isLoading ? 'none' : 'block'
    };

    const loadingStyle = {
        width: '150px',
        height: '150px',
        backgroundColor: '#f3f3f3',
        display: isLoading ? 'block' : 'none',
        textAlign: 'center',
    };

    if (media.type === "image") {
        return (
            <div style={mediaStyle} onClick={() => onClick(media)}>
                <div style={loadingStyle}>Loading...</div>
                <img
                    src={"http://192.168.1.69:8000/api/media/" + media.id}
                    alt={media.id}
                    style={imageStyle}
                    onLoad={() => setIsLoading(false)}
                    onError={() => setIsLoading(false)}
                />
                <p><center>{media.similarity.toFixed(4)}</center></p>
            </div>
        );
    }
};

const Modal = ({ media, onClose }) => {
    const modalStyle = {
        position: 'fixed',
        top: '0',
        left: '0',
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
    };

    const contentStyle = {
        backgroundColor: 'white',
        padding: '20px',
        maxWidth: '90%',
        maxHeight: '90%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
    };

    const imgStyle = {
        maxWidth: '100%',
        maxHeight: '80vh', // limit the maximum height to 80% of the viewport height
        objectFit: 'contain',
        margin: 'auto', // centers the image horizontally
    };

    const closeButtonStyle = {
        marginTop: '10px',
    };

    useEffect(() => {
        const handleKeyDown = (event) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };

        window.addEventListener('keydown', handleKeyDown);

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
        };
    }, [onClose]);

    return (
        <div style={modalStyle} onClick={onClose}>
            <div style={contentStyle} onClick={(event) => event.stopPropagation()}>
                <img
                    src={"http://192.168.1.69:8000/api/media/" + media.id}
                    alt={media.id}
                    style={imgStyle}
                />
                <button style={closeButtonStyle} onClick={onClose}>Close</button>
            </div>
        </div>
    );
};

const MediaGallery = ({ allMedia }) => {
    const [numPerPage, setNumPerPage] = useState(10);
    const [currentPage, setCurrentPage] = useState(1);
    const totalPages = Math.ceil(allMedia.length / numPerPage);

    const [isModalVisible, setModalVisible] = useState(false);
    const [modalContent, setModalContent] = useState(null);

    const goToPage = (page) => {
        const pageNumber = Math.max(1, Math.min(page, totalPages));
        setCurrentPage(pageNumber);
    };

    const handleMediaClick = (media) => {
        setModalContent(media);
        setModalVisible(true);
    };
    const currentMedia = allMedia.slice(
        (currentPage - 1) * numPerPage,
        currentPage * numPerPage
    );

    const galleryStyle = {
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'center',
        alignItems: 'center'
    };

    useEffect(() => {
        setCurrentPage(1);
    }, [allMedia]);

    return (
        <div>
            <div className="pagination">
                <button onClick={() => goToPage(currentPage - 1)} disabled={currentPage === 1}>Prev</button>
                <input
                    type="number"
                    value={currentPage}
                    onChange={(e) => goToPage(parseInt(e.target.value, 10))}
                    min="1"
                    max={totalPages}
                />
                <button onClick={() => goToPage(currentPage + 1)} disabled={currentPage === totalPages}>Next</button>
                <br />
                <label>Items per page: </label>
                <input
                    type="number"
                    value={numPerPage}
                    onChange={(e) => setNumPerPage(parseInt(e.target.value, 10))}
                    min="1"
                />
            </div>
            <div className="gallery" style={galleryStyle}>
                {currentMedia.map((media) => (
                    <Media key={media.id} media={media} onClick={() => handleMediaClick(media)} />
                ))}
            </div>
            {isModalVisible && <Modal media={modalContent} onClose={() => setModalVisible(false)} />}
        </div>
    );
};

export default MediaGallery;
