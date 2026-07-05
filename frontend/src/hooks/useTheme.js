import { useState, useLayoutEffect } from 'react';

/**
 * useTheme (Custom Hook)
 * ----------------------
 * Manages the global light/dark mode state for the application.
 * It persists the user's preference to `localStorage` so it survives page reloads,
 * and actively applies the `data-theme` attribute to the root `<html>` element
 * to trigger the CSS variable swaps. It also dynamically updates the browser favicon.
 */
export function useTheme() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem('theme') || 'light'
  );

  useLayoutEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    
    // Update favicon
    const favicon = document.getElementById('favicon');
    if (favicon) {
      favicon.href = theme === 'light' ? '/media/icons/light.png' : '/media/icons/dark.png';
    }
  }, [theme]);

  const toggle = () => setTheme((t) => (t === 'light' ? 'dark' : 'light'));

  return { theme, toggle };
}
