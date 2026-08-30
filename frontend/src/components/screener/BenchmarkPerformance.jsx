import { DollarSign, Info } from "lucide-react";
import GenericTabs from "../common/GenericTabs";
import DropdownSelector from "../common/DropdownSelector";
import "./BenchmarkPerformance.css";

/**
 * Displays a benchmark selection header with tabs (desktop) or a dropdown (mobile).
 * Shows the most recent data date in a tooltip.
 */
export default function BenchmarkPerformance({
  activeBenchmark,
  setActiveBenchmark,
  activeBenchmarkData,
  benchmarkDisplayNames,
}) {
  const recentDate =
    activeBenchmarkData?.dates?.length > 0
      ? new Date(
          activeBenchmarkData.dates[
            activeBenchmarkData.dates.length - 1
          ].replace(/-/g, "/"),
        ).toLocaleDateString()
      : "latest";

  return (
    <div className="benchmark-performance-section">
      <div className="benchmark-performance-header">
        <div className="benchmark-performance-title">
          <DollarSign className="icon-neutral" size={24} />
          <h2>Benchmark Performance</h2>
          <span
            data-tooltip={`Latest data metrics (as of ${recentDate})`}
            className="info-tooltip-container"
          >
            <Info size={16} />
          </span>
        </div>
      </div>

      <div className="benchmark-performance-toggles">
        <div className="hide-on-mobile tabs-wrapper">
          <GenericTabs
            tabs={Object.keys(benchmarkDisplayNames).map((bench) => ({
              id: bench,
              label: benchmarkDisplayNames[bench],
            }))}
            activeTab={activeBenchmark}
            onChange={setActiveBenchmark}
            containerClassName="tabs-container"
            tabClassName="tab-button"
            prefix={
              <span className="dropdown-selector-label">
                Choose a Benchmark:
              </span>
            }
          />
        </div>
        <DropdownSelector
          label="Choose a Benchmark:"
          options={Object.keys(benchmarkDisplayNames).map((bench) => ({
            value: bench,
            label: benchmarkDisplayNames[bench],
          }))}
          value={activeBenchmark}
          onChange={setActiveBenchmark}
          containerClassName="show-on-mobile dropdown-selector-container"
        />
      </div>
    </div>
  );
}
