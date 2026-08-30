import { useState, useEffect, useCallback, useMemo } from "react";
import { ChartNoAxesCombined as MarketOverviewIcon, Info } from "lucide-react";
import LineChart from "../charts/LineChart";
import CandlestickChart from "../charts/CandlestickChart";
import HeatmapChart from "../charts/HeatmapChart";
import GenericTabs from "../common/GenericTabs";
import ScreenerNavSlider from "../charts/ScreenerNavSlider";
import DropdownSelector from "../common/DropdownSelector";
import "./MarketCharts.css";

/**
 * Manages the display of market performance charts (Line, Candlestick, Heatmap).
 * Handles chart type switching, viewport resizing, and zoom states.
 */
export default function MarketCharts({
  benchmarkData,
  chartType,
  groupBySector,
  theme,
  chartTabs,
  setChartType,
  setGroupBySector,
}) {
  const [isMobileZoom, setIsMobileZoom] = useState(window.innerWidth <= 768);
  const [viewStates, setViewStates] = useState({});

  /**
   * Calculates the initial time range for charts.
   * On mobile, defaults to a 6-month view; on desktop, defaults to 1 year.
   */
  const getInitialViewState = useCallback(() => {
    if (benchmarkData?.dates?.length > 0) {
      const minTs = new Date(
        benchmarkData.dates[0].replace(/-/g, "/"),
      ).getTime();
      const maxTs = new Date(
        benchmarkData.dates[benchmarkData.dates.length - 1].replace(/-/g, "/"),
      ).getTime();
      const isMobile =
        typeof window !== "undefined" && window.innerWidth <= 768;
      const defaultDays = isMobile ? 180 : 365;
      const initialMin = Math.max(minTs, maxTs - defaultDays * 86400000);
      return {
        min: initialMin,
        max: maxTs,
        absoluteMin: minTs,
        absoluteMax: maxTs,
        activeRange: isMobile ? "6M" : "1Y",
      };
    }
    return null;
  }, [benchmarkData]);

  const viewState = useMemo(
    () => viewStates[chartType] || getInitialViewState(),
    [viewStates, chartType, getInitialViewState],
  );

  useEffect(() => {
    const handleResize = () => {
      const mobileZoom = window.innerWidth <= 768;
      setIsMobileZoom((prev) => {
        if (prev !== mobileZoom) {
          return mobileZoom;
        }
        return prev;
      });
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const handleViewChange = useCallback(
    (newView) => {
      setViewStates((prev) => ({
        ...prev,
        [chartType]: {
          ...(prev[chartType] || getInitialViewState()),
          ...newView,
        },
      }));
    },
    [chartType, getInitialViewState],
  );

  const handleResetView = useCallback(() => {
    setViewStates((prev) => {
      const current = prev[chartType] || getInitialViewState();
      if (!current) return prev;
      const isMobile =
        typeof window !== "undefined" && window.innerWidth <= 768;
      const defaultDays = isMobile ? 180 : 365;
      const initialMin = Math.max(
        current.absoluteMin,
        current.absoluteMax - defaultDays * 86400000,
      );
      return {
        ...prev,
        [chartType]: {
          ...current,
          min: initialMin,
          max: current.absoluteMax,
          activeRange: isMobile ? "6M" : "1Y",
        },
      };
    });
  }, [chartType, getInitialViewState]);

  useEffect(() => {
    setViewStates({});
  }, [benchmarkData?.name]);

  useEffect(() => {
    if (benchmarkData?.dates?.length > 0) {
      handleResetView();
    }
  }, [isMobileZoom, handleResetView, benchmarkData?.dates?.length]);

  if (!benchmarkData) return null;

  const benchmarkDisplayNames = {
    "Dow 30": "DOW 30",
    "Nasdaq 100": "NASDAQ 100",
    "S&P 500": "S&P 500",
    "Russell 1000": "RUSSELL 1000",
  };
  const displayName =
    benchmarkDisplayNames[benchmarkData.name] || benchmarkData.name;
  const isPos = benchmarkData.change >= 0;

  const chartTitle = <h2>{displayName}</h2>;
  const recentDate =
    benchmarkData.dates && benchmarkData.dates.length > 0
      ? new Date(
          benchmarkData.dates[benchmarkData.dates.length - 1].replace(
            /-/g,
            "/",
          ),
        ).toLocaleDateString()
      : "latest";

  const activeTooltipText =
    chartType === "heatmap" ? "1-Day Return" : "1-Year Return";

  const chartSubtitlePrice = (
    <div className="benchmark-price-row active-benchmark-price-row-centered">
      <span className="benchmark-price">
        Most Recent Closed Price: ${parseFloat(benchmarkData.price).toFixed(2)}
      </span>
      <span className={`benchmark-change ${isPos ? "positive" : "negative"}`}>
        {isPos ? "+" : ""}
        {parseFloat(benchmarkData.change).toFixed(2)}%
      </span>
      <span
        data-tooltip={activeTooltipText}
        className="info-tooltip-container info-tooltip-container-flex"
      >
        <Info size={16} />
      </span>
    </div>
  );

  let ChartComponent = null;
  if (chartType === "line" || !chartType) {
    ChartComponent = (
      <LineChart
        data={benchmarkData}
        isPositive={isPos}
        theme={theme}
        chartTitle={chartTitle}
        chartSubtitlePrice={chartSubtitlePrice}
        viewState={viewState}
      />
    );
  } else if (chartType === "candle") {
    ChartComponent = (
      <CandlestickChart
        data={benchmarkData}
        theme={theme}
        chartTitle={chartTitle}
        chartSubtitlePrice={chartSubtitlePrice}
        viewState={viewState}
      />
    );
  } else if (chartType === "heatmap") {
    ChartComponent = (
      <HeatmapChart
        data={benchmarkData}
        groupBySector={groupBySector}
        theme={theme}
        chartTitle={chartTitle}
        chartSubtitlePrice={chartSubtitlePrice}
      />
    );
  }

  const renderSlider = () => {
    if (chartType === "heatmap" || !viewState) return null;
    return (
      <div className="slider-container">
        <ScreenerNavSlider
          data={benchmarkData}
          theme={theme}
          viewState={viewState}
          onViewChange={handleViewChange}
          onReset={handleResetView}
        />
      </div>
    );
  };

  return (
    <div className="market-charts-section">
      <div className="screener-table-header screener-performance-header subsection-header">
        <div className="screener-table-title">
          <MarketOverviewIcon className="icon-neutral" size={20} />
          <h3>Market Charts</h3>
          <span
            data-tooltip={`Latest data metrics (as of ${recentDate})`}
            className="info-tooltip-container"
          >
            <Info size={16} />
          </span>
        </div>
      </div>

      <div className="screener-chart-toggles">
        <div className="hide-on-mobile tabs-wrapper">
          <GenericTabs
            tabs={chartTabs}
            activeTab={chartType}
            onChange={setChartType}
            containerClassName="tabs-container"
            tabClassName="tab-button"
            prefix={
              <span className="dropdown-selector-label">
                Choose a Chart Type:
              </span>
            }
          />
        </div>
        <DropdownSelector
          label="Choose a Chart Type:"
          options={chartTabs.map((tab) => ({
            value: tab.id,
            label:
              tab.id === "line"
                ? "Line Chart"
                : tab.id === "candle"
                  ? "Candlestick Chart"
                  : "Heatmap",
          }))}
          value={chartType}
          onChange={setChartType}
          containerClassName="show-on-mobile dropdown-selector-container"
        />
        {chartType === "heatmap" && (
          <div className="heatmap-radio-group">
            <span className="dropdown-selector-label">Group by Sector:</span>
            <label className="heatmap-radio-label">
              <input
                type="radio"
                name="groupBySector"
                checked={groupBySector}
                onChange={() => setGroupBySector(true)}
                className="heatmap-radio-input"
              />
              Yes
            </label>
            <label className="heatmap-radio-label">
              <input
                type="radio"
                name="groupBySector"
                checked={!groupBySector}
                onChange={() => setGroupBySector(false)}
                className="heatmap-radio-input"
              />
              No
            </label>
          </div>
        )}
      </div>

      <div className="benchmark-card active-benchmark-card active-benchmark-card-container">
        <div className="mini-chart-wrapper">
          <div
            className={`large-chart-container ${chartType === "heatmap" ? "heatmap-mode" : ""}`}
          >
            {ChartComponent}
          </div>
          {renderSlider()}
        </div>
      </div>
    </div>
  );
}
