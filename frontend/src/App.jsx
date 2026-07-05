/**
 * App.jsx
 * -------
 * The root component of the MarketLens React application.
 * Manages global application state, orchestrates the Server-Sent Events (SSE) connection
 * for the live prediction pipeline, and renders the primary UI layout including the Header,
 * SearchBar, Loading overlays, and the main Dashboard component.
 */
import { useState, useRef, useCallback } from 'react';
import { useTheme } from './hooks/useTheme';
import Header from './components/Header';
import SearchBar from './components/SearchBar';
import Loader from './components/Loader';
import ErrorMessage from './components/ErrorMessage';
import Dashboard from './components/Dashboard';
import NewsModal from './components/NewsModal';

export default function App() {
  const { theme, toggle: toggleTheme } = useTheme();

  // Application state
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [steps, setSteps] = useState([]);
  const [isLoaderVisible, setIsLoaderVisible] = useState(false);
  const [isLoaderFadingOut, setIsLoaderFadingOut] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [isFadeIn, setIsFadeIn] = useState(false);
  const [modalArticle, setModalArticle] = useState(null);

  // SSE step timer
  const timerIntervalRef = useRef(null);
  const activeStepIdRef = useRef(null);

  const clearLoading = useCallback(() => {
    setIsLoading(false);
    setIsLoaderVisible(false);
    setIsLoaderFadingOut(false);
    clearInterval(timerIntervalRef.current);
  }, []);

  const completeActiveStep = useCallback(() => {
    // Snapshot the ID synchronously so the functional setSteps update below
    // always closes over the OLD step's ID — not the one that gets written to
    // activeStepIdRef.current immediately after this call.
    const completingId = activeStepIdRef.current;
    if (!completingId) return;
    setSteps((prev) =>
      prev.map((s) =>
        s.id === completingId ? { ...s, status: 'completed' } : s
      )
    );
  }, []);

  const fetchPrediction = useCallback((ticker) => {
    if (!ticker) {
      // Clear signal from SearchBar
      setResult(null);
      setError('');
      setProgress(0);
      setSteps([]);
      return;
    }

    const upperTicker = ticker.toUpperCase();
    if (!upperTicker) {
      setError('Please enter a ticker symbol.');
      return;
    }

    // Strict validation to prevent SSRF / Path Traversal / Injection
    if (!/^[A-Z0-9.-]+$/.test(upperTicker)) {
      setError('Invalid ticker format. Only alphanumeric characters, dots, and hyphens are allowed.');
      return;
    }

    // Reset state for new fetch
    setError('');
    setResult(null);
    setProgress(0);
    setSteps([]);
    setIsLoading(true);
    setIsLoaderVisible(true);
    setIsLoaderFadingOut(false);
    clearInterval(timerIntervalRef.current);
    activeStepIdRef.current = null;

    try {
      const safeTicker = encodeURIComponent(upperTicker);
      const eventSource = new EventSource(`/predict_stream/${safeTicker}`);

      eventSource.onmessage = (e) => {
        const data = JSON.parse(e.data);

        if (data.status === 'error') {
          eventSource.close();
          clearInterval(timerIntervalRef.current);
          setError(data.error || 'An unknown error occurred.');
          clearLoading();
          return;
        }

        if (data.progress !== undefined) {
          setProgress(data.progress);
        }

        if (data.status === 'processing' && data.step) {
          // Complete the previous step
          completeActiveStep();

          // Create new active step
          const stepId = `step-${Date.now()}`;
          activeStepIdRef.current = stepId;

          setSteps((prev) => [
            ...prev,
            { id: stepId, label: data.step, status: 'active', timer: 0.0 },
          ]);

          // Start ticking the timer for this step
          clearInterval(timerIntervalRef.current);
          timerIntervalRef.current = setInterval(() => {
            setSteps((prev) =>
              prev.map((s) =>
                s.id === stepId ? { ...s, timer: parseFloat((s.timer + 0.1).toFixed(1)) } : s
              )
            );
          }, 100);
        }

        if (data.status === 'complete') {
          eventSource.close();
          clearInterval(timerIntervalRef.current);
          completeActiveStep();

          // Hold on 100% briefly so the user sees completion
          setTimeout(() => {
            setIsLoaderFadingOut(true);

            setTimeout(() => {
              clearLoading();
              setResult(data.result);
              setIsFadeIn(true);
              setTimeout(() => setIsFadeIn(false), 500);
            }, 350); // wait for fade-out animation
          }, 800); // 800ms hold on 100%
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        clearInterval(timerIntervalRef.current);
        setError('Connection to server lost. Please try again.');
        clearLoading();
      };
    } catch (err) {
      clearInterval(timerIntervalRef.current);
      setError(err.message);
      clearLoading();
    }
  }, [clearLoading, completeActiveStep]);

  const handleOpenModal = useCallback((article) => {
    setModalArticle(article);
  }, []);

  const handleCloseModal = useCallback(() => {
    setModalArticle(null);
  }, []);

  return (
    <div className="container">
      <Header theme={theme} onToggleTheme={toggleTheme} />

      <SearchBar onSearch={fetchPrediction} isLoading={isLoading} />

      <Loader
        visible={isLoaderVisible}
        progress={progress}
        steps={steps}
        isFadingOut={isLoaderFadingOut}
      />

      <ErrorMessage message={error} />

      {result && (
        <Dashboard
          data={result}
          theme={theme}
          isFadeIn={isFadeIn}
          onOpenModal={handleOpenModal}
        />
      )}

      <NewsModal article={modalArticle} onClose={handleCloseModal} />
    </div>
  );
}
