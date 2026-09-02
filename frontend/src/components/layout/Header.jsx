import { Moon, Sun } from "lucide-react";
import { Link } from "react-router-dom";
import "./Header.css";

// Application header component containing the logo and theme toggle switch
export default function Header({ theme, onToggleTheme }) {
  return (
    <div className="header-container">
      <div className="header-top">
        <Link to="/" className="logo-container">
          <img
            src={
              theme === "light"
                ? "/media/logos/light.png"
                : "/media/logos/dark.png"
            }
            alt="MarketLens Logo"
            className="app-logo"
          />
        </Link>
        <button
          id="themeToggle"
          className="theme-toggle-btn"
          title="Toggle Theme"
          onClick={onToggleTheme}
        >
          <div className="theme-slider">
            <div className={`theme-thumb ${theme}`}>
              {theme === "light" ? <Sun size={12} /> : <Moon size={12} />}
            </div>
          </div>
          <span className="theme-label">Theme</span>
        </button>
      </div>
    </div>
  );
}
