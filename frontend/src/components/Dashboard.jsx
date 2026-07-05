import { useState, useRef, useEffect } from 'react';
import SentimentTab from './tabs/SentimentTab';
import PriceTab from './tabs/PriceTab';
import DividendTab from './tabs/DividendTab';

const TABS = [
  { id: 'sentiment', label: 'Sentiment Analysis' },
  { id: 'price', label: 'Price Forecast' },
  { id: 'dividend', label: 'Dividend Forecast' },
];

export default function Dashboard({ data, theme, isFadeIn, onOpenModal }) {
  const [activeTab, setActiveTab] = useState('sentiment');
  const containerRef = useRef(null);

  // Trigger fade-in animation on initial render
  useEffect(() => {
    if (isFadeIn && containerRef.current) {
      containerRef.current.classList.add('fade-in');
      const timer = setTimeout(() => containerRef.current?.classList.remove('fade-in'), 500);
      return () => clearTimeout(timer);
    }
  }, [isFadeIn]);

  return (
    <div id="resultContainer" ref={containerRef}>
      <h2
        className="section-heading"
        style={{ marginTop: '10px', borderBottom: 'none', marginBottom: 0, paddingBottom: 0 }}
      >
        {data.Company_Name}{' '}
        <span style={{ color: 'var(--text-muted)', fontWeight: 600 }}>({data.Ticker})</span>
      </h2>

      <div className="tabs-container">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`tab-button${activeTab === tab.id ? ' active' : ''}`}
            data-tab={tab.id}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* All three tab panels are always mounted; CSS handles show/hide to preserve chart canvas */}
      <div style={{ position: 'relative' }}>
        <div className={`tab-content${activeTab === 'sentiment' ? ' active' : ''}`} id="tab-sentiment">
          <SentimentTab data={data} onOpenModal={onOpenModal} />
        </div>
        <div className={`tab-content${activeTab === 'price' ? ' active' : ''}`} id="tab-price">
          <PriceTab data={data} theme={theme} />
        </div>
        <div className={`tab-content${activeTab === 'dividend' ? ' active' : ''}`} id="tab-dividend">
          <DividendTab data={data} theme={theme} />
        </div>
      </div>
    </div>
  );
}
