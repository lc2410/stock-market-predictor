// Renders an error message if one exists; otherwise, returns null
export default function ErrorMessage({ message }) {
  if (!message) return null;
  return (
    <div id="errorContainer" className="error-message">
      Error: {message}
    </div>
  );
}
