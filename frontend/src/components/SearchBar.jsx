import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * Search bar with debounced autocomplete, clear button, and forecast trigger.
 */
export default function SearchBar({ onSearch, isLoading }) {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const debounceRef = useRef(null);
  const latestSearchIdRef = useRef(0);
  const inputRef = useRef(null);

  // Clear suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (
        inputRef.current &&
        !inputRef.current.parentElement.contains(e.target)
      ) {
        setSuggestions([]);
      }
    };
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const fetchSuggestions = useCallback(async (value) => {
    const currentId = ++latestSearchIdRef.current;
    try {
      const safeValue = encodeURIComponent(value);
      const res = await fetch(`/search/${safeValue}`);
      const data = await res.json();
      if (currentId !== latestSearchIdRef.current) return; // Stale response
      setSuggestions(data.length > 0 ? data : [{ symbol: null, name: 'No results found' }]);
    } catch {
      // Silently fail — don't break the UI
    }
  }, []);

  const handleInput = (e) => {
    const value = e.target.value;
    setQuery(value);
    setSuggestions([]);
    clearTimeout(debounceRef.current);
    if (!value.trim() || isLoading) return;
    debounceRef.current = setTimeout(() => fetchSuggestions(value.trim()), 300);
  };

  const handleClear = () => {
    setQuery('');
    setSuggestions([]);
    latestSearchIdRef.current++;
    onSearch(null); // Signal parent to clear display
    inputRef.current?.focus();
  };

  const handleSelect = (symbol) => {
    if (!symbol) return;
    clearTimeout(debounceRef.current);
    latestSearchIdRef.current++;
    setQuery(symbol);
    setSuggestions([]);
    onSearch(symbol);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      clearTimeout(debounceRef.current);
      latestSearchIdRef.current++;
      setSuggestions([]);
      onSearch(query.trim().toUpperCase());
    }
  };

  const handleSubmit = () => {
    clearTimeout(debounceRef.current);
    latestSearchIdRef.current++;
    setSuggestions([]);
    onSearch(query.trim().toUpperCase());
  };

  // Expose a way for the parent (App) to set the query value back after clear
  // The App passes the ticker back via a controlled pattern — handled by onSearch(null) clearing.

  return (
    <div className="input-group">
      <div className="input-wrapper">
        <input
          ref={inputRef}
          id="tickerInput"
          type="text"
          placeholder="e.g., AAPL, VOO, MSFT"
          autoFocus
          autoComplete="off"
          value={query}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
        />
        {query.length > 0 && !isLoading && (
          <button
            id="clearSearchBtn"
            title="Clear search"
            onClick={handleClear}
            style={{ display: 'block' }}
          >
            &times;
          </button>
        )}
        {suggestions.length > 0 && (
          <div id="autocompleteResults" className="autocomplete-items">
            {suggestions.map((item, idx) =>
              item.symbol ? (
                  <div
                    key={item.symbol + idx}
                    className="autocomplete-item"
                    data-symbol={item.symbol}
                    onClick={() => handleSelect(item.symbol)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') handleSelect(item.symbol);
                    }}
                    role="button"
                    tabIndex={0}
                  >
                  <span className="ac-sym">{item.symbol}</span>
                  <span className="ac-name">{item.name || ''}</span>
                </div>
              ) : (
                <div key={idx} className="autocomplete-item" style={{ color: '#999' }}>
                  No results found
                </div>
              )
            )}
          </div>
        )}
      </div>
      <button
        id="predictBtn"
        onClick={handleSubmit}
        disabled={isLoading}
        style={{ cursor: isLoading ? 'not-allowed' : '' }}
      >
        Get Forecast
      </button>
    </div>
  );
}
