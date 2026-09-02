import "./DropdownSelector.css";

// Reusable dropdown component with optional label and customizable styling
export default function DropdownSelector({
  options,
  value,
  onChange,
  label,
  containerClassName = "dropdown-selector-container",
  selectClassName = "dropdown-selector-select",
  labelClassName = "dropdown-selector-label",
  style,
}) {
  return (
    <div className={containerClassName} style={style}>
      {label && <span className={labelClassName}>{label}</span>}
      <select
        className={selectClassName}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}
