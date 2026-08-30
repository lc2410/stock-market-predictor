import { useState, useLayoutEffect } from 'react';

/**
 * Custom hook for toggling light/dark theme.
 * Syncs the selected theme with localStorage and updates DOM attributes.
 */
export function useTheme() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem('theme') || 'light'
  );

  useLayoutEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);

    const favicon = document.getElementById('favicon');
    if (favicon) {
      favicon.href = theme === 'light' ? '/media/icons/light.png' : '/media/icons/dark.png';
    }
  }, [theme]);

  const toggle = () => setTheme((t) => (t === 'light' ? 'dark' : 'light'));

  return { theme, toggle };
}
