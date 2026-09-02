import { useState, useEffect, useRef } from 'react';

/**
 * Custom hook to fetch and manage market screener data.
 * Includes artificial progress steps during initial fetch for smoother UX.
 */
export default function useScreenerData() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  // Loader states
  const [isLoaderFadingOut, setIsLoaderFadingOut] = useState(false);
  const [progress, setProgress] = useState(0);
  const [steps, setSteps] = useState([
    { id: 'step-1', label: 'Initializing homepage...', status: 'active', timer: 0.0 }
  ]);
  const timerRef = useRef(null);

  const [isFadeIn, setIsFadeIn] = useState(false);

  useEffect(() => {
    let ignore = false;
    timerRef.current = setInterval(() => {
      setSteps(prev => prev.map(s => s.id === 'step-1' ? { ...s, timer: Number.parseFloat((s.timer + 0.1).toFixed(1)) } : s));
      
      setProgress(p => {
        const remaining = 95 - p;
        return p + (remaining * 0.02);
      });
    }, 100);

    fetch(`/screener?t=${new Date().getTime()}`)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch screener data');
        return res.json();
      })
      .then(fetchedData => {
        if (ignore) return;
        clearInterval(timerRef.current);
        setSteps(prev => prev.map(s => s.id === 'step-1' ? { ...s, status: 'complete' } : s));
        setProgress(100);
        
        setTimeout(() => {
          if (ignore) return;
          setIsLoaderFadingOut(true);
          setTimeout(() => {
            if (ignore) return;
            setData(fetchedData);
            setLoading(false);
            setIsFadeIn(true);
            setTimeout(() => {
              if (!ignore) setIsFadeIn(false);
            }, 500);
          }, 400);
        }, 500);
      })
      .catch(err => {
        if (ignore) return;
        clearInterval(timerRef.current);
        setError(err.message);
        setLoading(false);
      });
      
    return () => {
      ignore = true;
      clearInterval(timerRef.current);
    };
  }, []);

  return { data, loading, error, isLoaderFadingOut, progress, steps, isFadeIn };
}
