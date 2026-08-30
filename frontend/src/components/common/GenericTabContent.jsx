// Wrapper component that handles displaying content based on the currently active tab
export default function GenericTabContent({
  id,
  activeTab,
  children,
  contentClassName = "tab-content",
  activeClassName = "active",
}) {
  return (
    <div
      className={`${contentClassName} ${activeTab === id ? activeClassName : ""}`}
      id={`tab-${id}`}
    >
      {children}
    </div>
  );
}
