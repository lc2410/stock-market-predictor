import { formatDate, formatMoney } from '../../utils/formatters';

/**
 * Unified scrollable data table for price and dividend history + forecast data.
 */
export default function DataTable({ title, dateHeader, histHeader, projHeader, rows }) {
  if (!rows || !rows.length) return null;

  const hasProj = rows.some((r) => r.proj !== null && r.proj !== undefined);
  const finalTitle = hasProj
    ? title
    : title.includes('Price')
    ? 'Closed Stock Price History'
    : 'Dividend Payout History';

  return (
    <>
      <h3 className="subsection-heading">{finalTitle}</h3>
      <div className="table-wrapper">
        <table className="glass-table">
          <thead>
            <tr>
              <th>{dateHeader}</th>
              <th>{histHeader}</th>
              {hasProj && <th>{projHeader}</th>}
              {hasProj && <th>Expected Range</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, idx) => {
              const histStr =
                r.hist !== null && r.hist !== undefined ? (
                  <strong style={{ color: 'var(--chart-history)' }}>
                    {formatMoney(r.hist)}
                  </strong>
                ) : (
                  '–'
                );

              const projStr =
                hasProj && r.proj !== null && r.proj !== undefined ? (
                  <strong style={{ color: 'rgba(var(--brand-rgb), 1)' }}>
                    {formatMoney(r.proj)}
                  </strong>
                ) : hasProj ? (
                  '–'
                ) : null;

              const ciStr =
                hasProj &&
                r.lower !== null &&
                r.upper !== null &&
                r.lower !== undefined &&
                r.upper !== undefined
                  ? `${formatMoney(r.lower)} – ${formatMoney(r.upper)}`
                  : hasProj
                  ? '–'
                  : null;

              return (
                <tr key={idx}>
                  <td>{formatDate(r.date)}</td>
                  <td>{histStr}</td>
                  {hasProj && <td>{projStr}</td>}
                  {hasProj && (
                    <td style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
                      {ciStr}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
