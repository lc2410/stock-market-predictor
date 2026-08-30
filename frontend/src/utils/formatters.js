
/**
 * Utility functions for formatting strings and extracting CSS variables for charts.
 */
export const formatDate = (dateStr) => {
  if (!dateStr || dateStr === 'N/A') return 'N/A';
  const parts = dateStr.split('-');
  if (parts.length !== 3) return dateStr;
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${months[parseInt(parts[1], 10) - 1]} ${parseInt(parts[2], 10)}, ${parts[0]}`;
};

export const formatMoney = (val) =>
  typeof val === 'number' ? `$${val.toFixed(2)}` : 'N/A';

export const getThemeColors = () => {
  const style = getComputedStyle(document.body);
  return {
    brandRGB: style.getPropertyValue('--brand-rgb').trim(),
    history: style.getPropertyValue('--chart-history').trim(),
    grid: style.getPropertyValue('--chart-grid').trim(),
    text: style.getPropertyValue('--text-main').trim(),
  };
};

export const normalizeDate = (isoString) => {
  if (!isoString) return '';
  return new Date(isoString).toISOString().split('T')[0];
};
