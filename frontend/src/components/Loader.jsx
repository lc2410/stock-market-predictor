import { useEffect, useRef } from 'react';

/**
 * Progress loader shown during SSE streaming.
 * Accepts `steps` (array of { label, timer, status }) and `progress` (0-100).
 */
export default function Loader({ visible, progress, steps, isFadingOut }) {
  const containerRef = useRef(null);

  // Manage the fade-out class on the container
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.classList.toggle('fade-out', isFadingOut);
    }
  }, [isFadingOut]);

  return (
    <div
      ref={containerRef}
      id="loader"
      className={`loader-container${visible ? ' visible' : ''}`}
    >
      <div className="progress-header">
        <h3 id="progress-title">Preparing Results...</h3>
        <span id="progress-percentage">{progress}%</span>
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
          <div
            key={step.id}
            className={`progress-step ${step.status}`}
          >
            <div className="step-left">
              {step.status === 'active' ? (
                <span className="step-icon spinner-icon" />
              ) : (
                <span
                  className="step-icon"
                  style={{ color: 'var(--brand-success)' }}
                >
                  ✓
                </span>
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
