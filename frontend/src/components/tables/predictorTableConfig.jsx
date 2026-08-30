import { formatDate, formatMoney } from "../../utils/formatters";

/**
 * Generates table column configurations for the PredictorTable.
 * Includes date and historical columns by default.
 * Conditionally appends projection and expected range columns if projection data is available.
 */
export const getPredictorTableColumns = (
  dateHeader,
  histHeader,
  projHeader,
  hasProj,
) => {
  const columns = [
    {
      header: dateHeader,
      key: "date",
      render: (r) => formatDate(r.date),
    },
    {
      header: histHeader,
      key: "hist",
      render: (r) => {
        return r.hist !== null && r.hist !== undefined ? (
          <strong className="data-table-hist-val">{formatMoney(r.hist)}</strong>
        ) : (
          "–"
        );
      },
    },
  ];

  if (hasProj) {
    columns.push({
      header: projHeader,
      key: "proj",
      render: (r) => {
        return r.proj !== null && r.proj !== undefined ? (
          <strong className="data-table-proj-val">{formatMoney(r.proj)}</strong>
        ) : (
          "–"
        );
      },
    });

    columns.push({
      header: "Expected Range",
      key: "range",
      cellClassName: () => "",
      render: (r) => {
        const hasBounds =
          r.lower !== null &&
          r.upper !== null &&
          r.lower !== undefined &&
          r.upper !== undefined;
        if (hasBounds) {
          return (
            <span className="data-table-range-val">
              {formatMoney(r.lower)} – {formatMoney(r.upper)}
            </span>
          );
        }
        return <span className="data-table-range-val">–</span>;
      },
    });
  }

  return columns;
};
