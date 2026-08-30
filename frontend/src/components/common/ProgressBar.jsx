import "./ProgressBar.css";

// Simple progress bar component with optional gradient styling
export default function ProgressBar({ pctValue, gradient = false }) {
  return (
    <div className="progress-bar-container">
      <div
        className={`progress-bar-fill ${gradient ? "gradient" : "solid"}`}
        style={{ width: `${pctValue}%` }}
      />
    </div>
  );
}
