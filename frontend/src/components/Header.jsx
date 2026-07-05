/**
 * Header Component
 * ----------------
 * Renders the top navigation area of the application.
 * Displays the dynamic MarketLens logo (which swaps based on the active theme),
 * a brief app description, and the native Light/Dark mode toggle button.
 */
export default function Header({ theme, onToggleTheme }) {
  return (
    <div className="header-container">
      <div className="header-top">
        <img src={theme === 'light' ? '/media/logos/light.png' : '/media/logos/dark.png'} alt="MarketLens Logo" className="app-logo" />
        <button id="themeToggle" className="theme-btn" title="Toggle Theme" onClick={onToggleTheme}>
          <span className="theme-icon">{theme === 'light' ? '🌙' : '☀️'}</span>
          <span className="theme-label">{theme === 'light' ? 'Dark' : 'Light'}</span>
        </button>
      </div>
      <p>
        Generate comprehensive price forecasts, dividend projections, and/or
        market sentiment analysis for any public asset.
      </p>
    </div>
  );
}
