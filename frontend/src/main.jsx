import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Chart,
  Filler,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  TimeScale,
  Tooltip,
  Legend,
  Title,
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import annotationPlugin from 'chartjs-plugin-annotation';
import './index.css';
import App from './App.jsx';

// Register all Chart.js components globally so they are available to every chart
Chart.register(
  Filler,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  TimeScale,
  Tooltip,
  Legend,
  Title,
  annotationPlugin
);

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>
);
