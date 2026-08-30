import { useState } from "react";
import SentimentAnalysis from "../components/predictor/SentimentAnalysis";
import PriceForecast from "../components/predictor/PriceForecast";
import DividendForecast from "../components/predictor/DividendForecast";
import GenericTabs from "../components/common/GenericTabs";
import DropdownSelector from "../components/common/DropdownSelector";
import GenericTabContent from "../components/common/GenericTabContent";
import { Info } from "lucide-react";
import "./StockPredictorPage.css";

const TABS = [
  { id: "sentiment", label: "Sentiment Analysis" },
  { id: "price", label: "Price Forecast" },
  { id: "dividend", label: "Dividend Forecast" },
];

/**
 * Stock Predictor Page Component.
 * Renders the prediction results for a specific stock, split into tabs:
 * Sentiment Analysis, Price Forecast, and Dividend Forecast.
 */
export default function StockPredictorPage({
  data,
  theme,
  isFadeIn,
  onOpenModal,
}) {
  const [activeTab, setActiveTab] = useState("sentiment");

  const recentDate =
    data.Chart_History?.dates?.length > 0
      ? new Date(
          data.Chart_History.dates[data.Chart_History.dates.length - 1].replace(
            /-/g,
            "/",
          ),
        ).toLocaleDateString()
      : new Date().toLocaleDateString();

  return (
    <div id="resultContainer" className={isFadeIn ? "fade-in" : ""}>
      <h2 className="section-heading predictor-page-heading">
        {data.Company_Name}{" "}
        <span className="predictor-page-ticker">({data.Ticker})</span>
        <span
          data-tooltip={`Latest forecast and data metrics (as of ${recentDate})`}
          className="info-tooltip-container"
        >
          <Info size={16} />
        </span>
      </h2>

      <div className="hide-on-mobile tabs-wrapper predictor-tabs-wrapper">
        <GenericTabs
          tabs={TABS}
          activeTab={activeTab}
          onChange={setActiveTab}
          containerClassName="tabs-container"
          tabClassName="tab-button"
          prefix={<span className="dropdown-selector-label">Select View:</span>}
        />
      </div>
      <DropdownSelector
        label="Select View:"
        options={TABS.map((tab) => ({
          value: tab.id,
          label: tab.label,
        }))}
        value={activeTab}
        onChange={setActiveTab}
        containerClassName="show-on-mobile dropdown-selector-container predictor-mobile-dropdown"
      />

      <div className="predictor-tab-container">
        <GenericTabContent id="sentiment" activeTab={activeTab}>
          <SentimentAnalysis data={data} onOpenModal={onOpenModal} />
        </GenericTabContent>

        <GenericTabContent id="price" activeTab={activeTab}>
          <PriceForecast
            key={data.Ticker || "price"}
            data={data}
            theme={theme}
          />
        </GenericTabContent>

        <GenericTabContent id="dividend" activeTab={activeTab}>
          <DividendForecast data={data} theme={theme} />
        </GenericTabContent>
      </div>
    </div>
  );
}
