/**
 * Configuration for market screener tables.
 * Defines the layout and data accessors for different market indicators (e.g., top gainers, most actives).
 */
export const TABLE_CONFIGS = [
  {
    id: "day_gainers",
    titleSuffix: "Top Gainers",
    description: "Stocks with the highest price gain",
    type: "movers",
    orderBy: "change",
    getData: (data, benchmark) => data.market_movers?.[benchmark]?.day_gainers,
  },
  {
    id: "day_losers",
    titleSuffix: "Top Losers",
    description: "Stocks with the highest price loss",
    type: "movers",
    orderBy: "change",
    getData: (data, benchmark) => data.market_movers?.[benchmark]?.day_losers,
  },
  {
    id: "most_actives",
    titleSuffix: "Most Active",
    description: "Stocks with the highest trading volume",
    type: "volume",
    orderBy: "volume",
    getData: (data, benchmark) => data.market_movers?.[benchmark]?.most_actives,
  },
  {
    id: "biggest_dividends",
    titleSuffix: "Biggest Dividend Yields",
    description:
      "Stocks with the biggest dividend yield given their last payout",
    type: "dividend",
    orderBy: "dividend_yield",
    getData: (data, benchmark) =>
      data.custom_scans?.[benchmark]?.biggest_dividends,
  },
  {
    id: "new_high",
    titleSuffix: "New 52-Week Highs",
    description: "Stocks with a new 52 week high price",
    type: "new_high",
    orderBy: "breakout_high_pct",
    getData: (data, benchmark) => data.custom_scans?.[benchmark]?.new_high,
  },
  {
    id: "new_low",
    titleSuffix: "New 52-Week Lows",
    description: "Stocks with a new 52 week low price",
    type: "new_low",
    orderBy: "breakout_low_pct",
    getData: (data, benchmark) => data.custom_scans?.[benchmark]?.new_low,
  },
  {
    id: "most_volatile",
    titleSuffix: "Most Volatile",
    description: "Stocks with the widest high-to-low trading range",
    type: "volatility",
    orderBy: "volatility",
    getData: (data, benchmark) => data.custom_scans?.[benchmark]?.most_volatile,
  },
  {
    id: "overbought",
    titleSuffix: "Overbought",
    description:
      "Stocks with an extreme price increase over the past 2 weeks, calculated by RSI(14) indicator",
    type: "overbought",
    orderBy: "rsi",
    getData: (data, benchmark) => data.custom_scans?.[benchmark]?.overbought,
  },
  {
    id: "oversold",
    titleSuffix: "Oversold",
    description:
      "Stocks with an extreme price decrease over the past 2 weeks, calculated by RSI(14) indicator",
    type: "oversold",
    orderBy: "rsi",
    getData: (data, benchmark) => data.custom_scans?.[benchmark]?.oversold,
  },
  {
    id: "unusual_volume",
    titleSuffix: "Unusual Volume",
    description: "Stocks with an unusually high volume",
    type: "unusual_volume",
    orderBy: "vol_change_pct",
    getData: (data, benchmark) =>
      data.custom_scans?.[benchmark]?.unusual_volume,
  },
];
