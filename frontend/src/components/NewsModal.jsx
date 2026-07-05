import { useEffect, useRef } from 'react';
import ReactDOM from 'react-dom';

/**
 * News article preview modal rendered as a React portal.
 * Features a fade-in/slide-up animation on open and
 * fade-out animation on close.
 */
export default function NewsModal({ article, onClose }) {
  const overlayRef = useRef(null);

  // Handle Escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    if (article) {
      document.addEventListener('keydown', handleKeyDown);
    }
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [article, onClose]);

  if (!article) return null;

  const handleOverlayClick = (e) => {
    if (e.target === overlayRef.current) onClose();
  };

  return ReactDOM.createPortal(
    <div
      ref={overlayRef}
      id="newsModal"
      className="modal-overlay"
      onClick={handleOverlayClick}
    >
      <div className="modal-content">
        <button id="closeModalBtn" className="modal-close" onClick={onClose}>
          &times;
        </button>
        <h2 id="modalTitle" className="modal-title">
          {article.title}
        </h2>
        <div className="modal-meta">
          <span>Read article on </span>
          <strong id="modalPublisher">{article.publisher}</strong>
          <a
            id="modalExternalLink"
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            title="Open external article"
            className="modal-export-link"
          >
            <svg
              className="export-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
              <polyline points="15 3 21 3 21 9" />
              <line x1="10" y1="14" x2="21" y2="3" />
            </svg>
          </a>
        </div>
        <div className="modal-body">
          <p id="modalSummary">{article.summary}</p>
        </div>
      </div>
    </div>,
    document.body
  );
}
