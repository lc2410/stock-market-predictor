import { useState, useCallback } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  useNavigate,
  useLocation,
} from "react-router-dom";
import { useTheme } from "./hooks/useTheme";
import usePredictorData from "./hooks/usePredictorData";
import Header from "./components/layout/Header";
import SearchBar from "./components/layout/SearchBar";
import Loader from "./components/layout/Loader";
import ErrorMessage from "./components/layout/ErrorMessage";
import NewsModal from "./components/layout/NewsModal";
import StockPredictorPage from "./pages/StockPredictorPage";
import MarketScreenerPage from "./pages/MarketScreenerPage";

/** 
 * Main application layout and routing.
 * Manages global state for theme, predictor data fetching, and search interactions.
 */
function AppContent() {
  const { theme, toggle: toggleTheme } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();

  const {
    fetchPrediction,
    cancelPrediction,
    clearPrediction,
    isLoading,
    progress,
    steps,
    isLoaderVisible,
    isLoaderFadingOut,
    error,
    result,
    isFadeIn,
    resolvedTicker,
  } = usePredictorData();

  const [modalArticle, setModalArticle] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");

  // Triggers data fetching and routing for a specific stock ticker
  const handleSearch = useCallback(
    (ticker) => {
      if (!ticker) {
        clearPrediction();
        setSearchQuery("");
        navigate("/");
        return;
      }
      setSearchQuery(ticker);
      navigate(`/predict/${encodeURIComponent(ticker.toUpperCase())}`);
      fetchPrediction(ticker);
    },
    [fetchPrediction, clearPrediction, navigate],
  );

  const handleCancel = useCallback(() => {
    cancelPrediction();
    setSearchQuery("");
    navigate("/");
  }, [cancelPrediction, navigate]);

  const handleOpenModal = useCallback((article) => {
    setModalArticle(article);
  }, []);

  const handleCloseModal = useCallback(() => {
    setModalArticle(null);
  }, []);

  return (
    <div
      className={`container ${location.pathname === "/" || location.pathname === "/screener" ? "screener-container" : ""}`}
    >
      <Header theme={theme} onToggleTheme={toggleTheme} />

      <SearchBar
        onSearch={handleSearch}
        onCancel={handleCancel}
        isLoading={isLoading}
        resolvedTicker={searchQuery || resolvedTicker}
      />

      {error && <ErrorMessage message={error} />}

      <Loader
        visible={isLoaderVisible}
        progress={progress}
        steps={steps}
        isFadingOut={isLoaderFadingOut}
        resolvedTicker={resolvedTicker}
        title={
          resolvedTicker
            ? `Analyzing ${resolvedTicker}...`
            : "Preparing Results..."
        }
      />

      <Routes>
        <Route
          path="/"
          element={
            <MarketScreenerPage
              theme={theme}
              onNewsClick={handleOpenModal}
              onTickerSearch={handleSearch}
            />
          }
        />
        <Route
          path="/screener"
          element={
            <MarketScreenerPage
              theme={theme}
              onNewsClick={handleOpenModal}
              onTickerSearch={handleSearch}
            />
          }
        />
        <Route
          path="/predict/:ticker"
          element={
            !isLoading && result ? (
              <StockPredictorPage
                data={result}
                theme={theme}
                isFadeIn={isFadeIn}
                onOpenModal={handleOpenModal}
              />
            ) : null
          }
        />
      </Routes>

      {modalArticle && (
        <NewsModal
          article={modalArticle}
          onClose={handleCloseModal}
          theme={theme}
        />
      )}
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}
