import "./GenericTabs.css";

// Renders a list of clickable tab buttons used for switching between different sections or views
export default function GenericTabs({
  tabs,
  activeTab,
  onChange,
  containerClassName = "tabs-container",
  tabClassName = "tab-button",
  activeClassName = "active",
  containerStyle = {},
  prefix,
  children,
}) {
  return (
    <div className={containerClassName} style={containerStyle}>
      {prefix}
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={`${tabClassName} ${activeTab === tab.id ? activeClassName : ""} ${tab.styleClassName || ""}`}
          onClick={() => onChange(tab.id)}
          data-tooltip={tab.tooltip}
          style={tab.style}
        >
          {tab.label}
        </button>
      ))}
      {children}
    </div>
  );
}
