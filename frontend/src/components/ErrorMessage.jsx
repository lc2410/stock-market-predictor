export default function ErrorMessage({ message }) {
  if (!message) return null;
  return (
    <div id="errorContainer" className="error-message">
      Error: {message}
    </div>
  );
}
