import { useState, useRef, useCallback } from 'react';

/**
 * Custom hook to manage stock prediction state and Server-Sent Events (SSE) streaming.
 * Handles the lifecycle of a prediction request, including loading steps and result processing.
 */
export default function usePredictorData() {
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [steps, setSteps] = useState([]);
  const [isLoaderVisible, setIsLoaderVisible] = useState(false);
  const [isLoaderFadingOut, setIsLoaderFadingOut] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [isFadeIn, setIsFadeIn] = useState(false);
  const [resolvedTicker, setResolvedTicker] = useState(null);

  const timerIntervalRef = useRef(null);
  const activeStepIdRef = useRef(null);
  const eventSourceRef = useRef(null);

  const clearLoading = useCallback(() => {
    setIsLoading(false);
    setIsLoaderVisible(false);
    setIsLoaderFadingOut(false);
    clearInterval(timerIntervalRef.current);
  }, []);

  const clearPrediction = useCallback(() => {
    setResult(null);
    setError('');
    setProgress(0);
    setSteps([]);
    setResolvedTicker(null);
  }, []);

  const cancelPrediction = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    clearLoading();
    setError('');
  }, [clearLoading]);

  const completeActiveStep = useCallback(() => {
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
      clearPrediction();
      return;
    }

    const upperTicker = ticker.toUpperCase();
    if (!upperTicker) {
      setError('Please enter a ticker symbol.');
      return;
    }

    if (!/^[A-Z0-9.-]+$/.test(upperTicker)) {
      setError('Invalid ticker format. Only alphanumeric characters, dots, and hyphens are allowed.');
      return;
    }

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    clearPrediction();
    setIsLoading(true);
    setIsLoaderVisible(true);
    setIsLoaderFadingOut(false);
    clearInterval(timerIntervalRef.current);
    activeStepIdRef.current = null;

    try {
      const safeTicker = encodeURIComponent(upperTicker);
      // Initialize Server-Sent Events (SSE) connection for real-time prediction updates
      const eventSource = new EventSource(`/predict_stream/${safeTicker}`);
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (e) => {
        if (eventSource !== eventSourceRef.current) return;
        const data = JSON.parse(e.data);

        if (data.status === 'error') {
          eventSource.close();
          eventSourceRef.current = null;
          clearInterval(timerIntervalRef.current);
          setError(data.error || 'An unknown error occurred.');
          clearLoading();
          return;
        }

        if (data.progress !== undefined) {
          setProgress(data.progress);
        }

        if (data.resolvedTicker) {
          setResolvedTicker(data.resolvedTicker);
        }

        if (data.status === 'processing' && data.step) {
          completeActiveStep();
          const stepId = `step-${Date.now()}`;
          activeStepIdRef.current = stepId;

          setSteps((prev) => [
            ...prev,
            { id: stepId, label: data.step, status: 'active', timer: 0.0 },
          ]);

          clearInterval(timerIntervalRef.current);
          timerIntervalRef.current = setInterval(() => {
            setSteps((prev) =>
              prev.map((s) =>
                s.id === stepId ? { ...s, timer: Number.parseFloat((s.timer + 0.1).toFixed(1)) } : s
              )
            );
          }, 100);
        }

        if (data.status === 'complete') {
          eventSource.close();
          eventSourceRef.current = null;
          clearInterval(timerIntervalRef.current);
          completeActiveStep();

          setTimeout(() => {
            setIsLoaderFadingOut(true);
            setTimeout(() => {
              clearLoading();
              setResult(data.result);
              setIsFadeIn(true);
              setTimeout(() => setIsFadeIn(false), 500);
            }, 350); 
          }, 800); 
        }
      };

      eventSource.onerror = () => {
        if (eventSource !== eventSourceRef.current) return;
        eventSource.close();
        eventSourceRef.current = null;
        clearInterval(timerIntervalRef.current);
        setError('Connection to server lost. Please try again.');
        clearLoading();
      };
    } catch (err) {
      clearInterval(timerIntervalRef.current);
      setError(err.message);
      clearLoading();
    }
  }, [clearLoading, completeActiveStep, clearPrediction]);

  return {
    fetchPrediction,
    cancelPrediction,
    clearPrediction,
    isLoading,
    progress,
    steps,
    isLoaderVisible,
    isLoaderFadingOut,
    error,
    result,
    isFadeIn,
    resolvedTicker
  };
}
