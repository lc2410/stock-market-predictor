import "./GenericTable.css";

// Reusable table component that flexibly renders columns and rows based on a configuration array
export default function GenericTable({
  columns,
  data,
  tableClassName = "glass-table",
  wrapperClassName = "table-wrapper",
  rowKey = (row, index) => index,
}) {
  if (!data || data.length === 0) return null;

  return (
    <div className={wrapperClassName}>
      <table className={tableClassName}>
        <thead>
          <tr>
            {columns.map((col, idx) => (
              <th key={idx} className={col.headerClassName || ""}>
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr key={rowKey(row, idx)}>
              {columns.map((col, colIdx) => {
                const content = col.render
                  ? col.render(row, idx)
                  : row[col.key];
                const cellClass =
                  typeof col.cellClassName === "function"
                    ? col.cellClassName(row, idx)
                    : col.cellClassName || "";

                return (
                  <td
                    key={colIdx}
                    className={cellClass}
                    data-label={col.header}
                  >
                    {content}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
