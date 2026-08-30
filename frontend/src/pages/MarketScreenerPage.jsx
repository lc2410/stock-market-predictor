import { useState } from "react";
import Loader from "../components/layout/Loader";
import TopHeadlines from "../components/screener/TopHeadlines";
import MarketCharts from "../components/screener/MarketCharts";
import MarketDataTables from "../components/screener/MarketDataTables";
import "./MarketScreenerPage.css";
import {
  CandlestickChart as CandlestickIcon,
  Blocks as BlocksIcon,
  LineChart as LineChartIcon,
} from "lucide-react";
import BenchmarkPerformance from "../components/screener/BenchmarkPerformance";
import useScreenerData from "../hooks/useScreenerData";

/**
 * Market Screener Page Component.
 * Displays overall market benchmarks, charts (Line, Candlestick, Heatmap),
 * data tables (gainers, losers, etc.), and top market news.
 */
export default function MarketScreenerPage({
  theme,
  onNewsClick,
  onTickerSearch,
}) {
  const { data, loading, error, isLoaderFadingOut, progress, steps, isFadeIn } =
    useScreenerData();
  const [chartType, setChartType] = useState("line");
  const [groupBySector, setGroupBySector] = useState(false);
  const [activeBenchmark, setActiveBenchmark] = useState("S&P 500");

  if (loading) {
    return (
      <div className={`screener-home ${theme} screener-loader-container`}>
        <Loader
          visible={true}
          progress={progress}
          steps={steps}
          isFadingOut={isLoaderFadingOut}
          title="Loading Market Data..."
        />
      </div>
    );
  }

  if (error) {
    return (
      <div className="screener-error">Error loading screener data: {error}</div>
    );
  }

  if (!data) return null;

  const benchmarkDisplayNames = {
    "Dow 30": "DOW 30",
    "Nasdaq 100": "NASDAQ 100",
    "S&P 500": "S&P 500",
    "Russell 1000": "RUSSELL 1000",
  };

  const activeBenchmarkData =
    data.benchmarks?.find((b) => b.name === activeBenchmark) ||
    data.benchmarks?.[0];

  const chartTabs = [
    {
      id: "line",
      label: (
        <span className="chart-tab-label">
          <LineChartIcon size={20} /> Line Chart
        </span>
      ),
    },
    {
      id: "candle",
      label: (
        <span className="chart-tab-label">
          <CandlestickIcon size={20} /> Candlestick Chart
        </span>
      ),
    },
    {
      id: "heatmap",
      label: (
        <span className="chart-tab-label">
          <BlocksIcon size={20} /> Heatmap
        </span>
      ),
    },
  ];

  return (
    <div className={`screener-home ${theme} ${isFadeIn ? "fade-in" : ""}`}>
      <BenchmarkPerformance
        activeBenchmark={activeBenchmark}
        setActiveBenchmark={setActiveBenchmark}
        activeBenchmarkData={activeBenchmarkData}
        benchmarkDisplayNames={benchmarkDisplayNames}
      />

      <MarketCharts
        benchmarkData={activeBenchmarkData}
        chartType={chartType}
        groupBySector={groupBySector}
        theme={theme}
        chartTabs={chartTabs}
        setChartType={setChartType}
        setGroupBySector={setGroupBySector}
      />

      <MarketDataTables
        data={data}
        activeBenchmark={activeBenchmark}
        displayName={
          benchmarkDisplayNames[activeBenchmarkData?.name] ||
          activeBenchmarkData?.name ||
          activeBenchmark
        }
        recentDate={
          activeBenchmarkData?.dates?.length > 0
            ? new Date(
                activeBenchmarkData.dates[
                  activeBenchmarkData.dates.length - 1
                ].replace(/-/g, "/"),
              ).toLocaleDateString()
            : "latest"
        }
        onTickerSearch={onTickerSearch}
      />

      <div className="screener-headlines-container">
        <TopHeadlines headlines={data.headlines} onNewsClick={onNewsClick} />
      </div>
    </div>
  );
}
