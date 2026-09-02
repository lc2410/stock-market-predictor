import { useState } from "react";
import { Sheet as SheetIcon, Info } from "lucide-react";
import ScreenerTable from "../tables/ScreenerTable";
import { TABLE_CONFIGS } from "../tables/screenerTableConfig";
import GenericTabs from "../common/GenericTabs";
import DropdownSelector from "../common/DropdownSelector";
import "./MarketDataTables.css";

/**
 * Provides a UI for selecting and displaying various market metric tables.
 * Uses TABLE_CONFIGS to dynamically render different data views (e.g. top gainers).
 */
export default function MarketDataTables({
  data,
  activeBenchmark,
  displayName,
  recentDate,
  onTickerSearch,
}) {
  const [activeTableId, setActiveTableId] = useState("day_gainers");

  if (!data) return null;

  const activeConfig =
    TABLE_CONFIGS.find((c) => c.id === activeTableId) || TABLE_CONFIGS[0];
  const tableData = activeConfig.getData(data, activeBenchmark);

  return (
    <div className="market-data-tables-section">
      <div className="screener-table-header screener-tables-header subsection-header">
        <div className="screener-table-title">
          <SheetIcon className="icon-neutral" size={20} />
          <h3>Market Data Tables</h3>
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
            tabs={TABLE_CONFIGS.map((tab) => ({
              id: tab.id,
              label: tab.titleSuffix,
            }))}
            activeTab={activeTableId}
            onChange={setActiveTableId}
            containerClassName="tabs-container"
            tabClassName="tab-button"
            prefix={
              <span className="dropdown-selector-label">
                Choose a Metric Data Table:
              </span>
            }
          />
        </div>
        <DropdownSelector
          label="Choose a Metric Data Table:"
          options={TABLE_CONFIGS.map((tab) => ({
            value: tab.id,
            label: tab.titleSuffix,
          }))}
          value={activeTableId}
          onChange={setActiveTableId}
          containerClassName="show-on-mobile dropdown-selector-container"
        />
      </div>

      {tableData?.length > 0 ? (
        <ScreenerTable
          title={`${displayName} ${activeConfig.titleSuffix}`}
          description={activeConfig.description}
          data={tableData}
          type={activeConfig.type}
          orderBy={activeConfig.orderBy}
          onTickerSearch={onTickerSearch}
        />
      ) : (
        <div className="no-data screener-no-data">
          No available data for this metric
        </div>
      )}
    </div>
  );
}
