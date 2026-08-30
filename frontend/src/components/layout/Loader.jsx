import { useEffect, useRef, useState } from "react";
import "../common/Loader.css";

// Full-screen loader displaying progress percentage and individual step statuses
export default function Loader({
  visible,
  progress,
  steps,
  isFadingOut,
  title = "Preparing Results...",
}) {
  const containerRef = useRef(null);
  const [displayedProgress, setDisplayedProgress] = useState(0);

  // Smoothly animate the progress percentage to match the target progress
  useEffect(() => {
    let animationFrameId;
    let current = displayedProgress;

    if (progress === 0) {
      setDisplayedProgress(0);
      return;
    }

    const animate = () => {
      const diff = progress - current;
      if (Math.abs(diff) < 0.1) {
        current = progress;
        setDisplayedProgress(progress);
      } else {
        current += diff * 0.15;
        setDisplayedProgress(current);
        animationFrameId = requestAnimationFrame(animate);
      }
    };

    animationFrameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrameId);
  }, [progress]);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.classList.toggle("fade-out", isFadingOut);
    }
  }, [isFadingOut]);

  return (
    <div
      ref={containerRef}
      id="loader"
      className={`loader-container${visible ? " visible" : ""}`}
    >
      <div className="progress-header">
        <h3 id="progress-title">{title}</h3>
        <span id="progress-percentage">{Math.round(displayedProgress)}%</span>
      </div>
      <div className="progress-bar-bg">
        <div
          id="progress-bar-fill"
          className="progress-bar-fill"
          style={{ width: `${progress}%` }}
        />
      </div>
      <div className="progress-steps" id="progress-steps-container">
        {steps.map((step) => (
          <div key={step.id} className={`progress-step ${step.status}`}>
            <div className="step-left">
              {step.status === "active" ? (
                <span className="step-icon spinner-icon" />
              ) : (
                <span className="step-icon step-icon-success">✓</span>
              )}
              <span>{step.label}</span>
            </div>
            <span className="step-timer" id={step.id}>
              {step.timer.toFixed(1)}s
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
