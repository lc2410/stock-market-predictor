import { useState, useRef, useCallback, useEffect } from "react";
import "./SearchBar.css";

// Search input component with debounce and auto-suggestions
export default function SearchBar({
  onSearch,
  onCancel,
  isLoading,
  resolvedTicker,
}) {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const debounceRef = useRef(null);
  const latestSearchIdRef = useRef(0);
  const inputRef = useRef(null);

  // Sync with resolved ticker from backend if applicable
  useEffect(() => {
    if (resolvedTicker) {
      setQuery(resolvedTicker);
    }
  }, [resolvedTicker]);

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
    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  }, []);

  useEffect(() => {
    setSelectedIndex(suggestions.length > 0 && suggestions[0].symbol ? 0 : -1);
  }, [suggestions]);

  // Fetch ticker suggestions from the server based on user input
  const fetchSuggestions = useCallback(async (value) => {
    if (!/^[a-zA-Z0-9.\-\s]+$/.test(value)) {
      setSuggestions([]);
      return;
    }

    const currentId = ++latestSearchIdRef.current;
    try {
      const safeValue = encodeURIComponent(value);
      const res = await fetch(`/search/${safeValue}`);
      const data = await res.json();
      if (currentId !== latestSearchIdRef.current) return;
      setSuggestions(
        data.length > 0 ? data : [{ symbol: null, name: "No results found" }],
      );
    } catch (error) {
      console.error("Search failed:", error);
    }
  }, []);

  const handleInput = (e) => {
    const value = e.target.value;
    setQuery(value);
    setSuggestions([]);
    latestSearchIdRef.current++;
    clearTimeout(debounceRef.current);
    if (!value.trim() || isLoading) return;
    debounceRef.current = setTimeout(() => fetchSuggestions(value.trim()), 300);
  };

  const handleClear = () => {
    setQuery("");
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
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) =>
        prev < suggestions.length - 1 ? prev + 1 : prev,
      );
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (
        selectedIndex >= 0 &&
        suggestions[selectedIndex]?.symbol
      ) {
        handleSelect(suggestions[selectedIndex].symbol);
      } else {
        clearTimeout(debounceRef.current);
        latestSearchIdRef.current++;
        setSuggestions([]);
        onSearch(query.trim().toUpperCase());
      }
    }
  };

  const handleSubmit = () => {
    clearTimeout(debounceRef.current);
    latestSearchIdRef.current++;
    setSuggestions([]);
    onSearch(query.trim().toUpperCase());
  };

  const handleCancelClick = () => {
    setQuery("");
    setSuggestions([]);
    onCancel();
  };

  return (
    <div className="input-group">
      <div className="input-wrapper">
        <input
          ref={inputRef}
          id="tickerInput"
          type="text"
          placeholder="Search for a ticker to gain potential market insights (i.e. AAPL, VOO, etc...)"
          autoFocus
          autoComplete="off"
          value={query}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          role="combobox"
          aria-autocomplete="list"
          aria-controls="autocompleteResults"
          aria-expanded={suggestions.length > 0}
          aria-activedescendant={
            selectedIndex >= 0 ? `suggestion-${selectedIndex}` : undefined
          }
        />
        {query.length > 0 && !isLoading && (
          <button
            id="clearSearchBtn"
            title="Clear search"
            onClick={handleClear}
          >
            &times;
          </button>
        )}
        {suggestions.length > 0 && (
          <div
            id="autocompleteResults"
            className="autocomplete-items"
            role="listbox"
          >
            {suggestions.map((item, idx) =>
              item.symbol ? (
                <div
                  key={item.symbol + idx}
                  id={`suggestion-${idx}`}
                  className={`autocomplete-item ${idx === selectedIndex ? "active" : ""}`}
                  data-symbol={item.symbol}
                  onClick={() => handleSelect(item.symbol)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ")
                      handleSelect(item.symbol);
                  }}
                  role="option"
                  aria-selected={idx === selectedIndex}
                  tabIndex={0}
                >
                  <span className="ac-sym">{item.symbol}</span>
                  <span className="ac-name">{item.name || ""}</span>
                </div>
              ) : (
                <div
                  key={idx}
                  className="autocomplete-item autocomplete-no-results"
                >
                  No results found
                </div>
              ),
            )}
          </div>
        )}
      </div>
      <button
        id="predictBtn"
        onClick={isLoading ? handleCancelClick : handleSubmit}
        className={isLoading ? "cancel-btn" : ""}
      >
        {isLoading ? "Cancel" : "Get Forecast"}
      </button>
    </div>
  );
}
