// frontend/src/index.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'; // Global styles
import App from './App'; // Your main App component (will be the "home" page)
import TestSubpage from './components/TestSubpage'; // The new subpage component
import reportWebVitals from './reportWebVitals';

import { BrowserRouter, Routes, Route, Link as RouterLink } from 'react-router-dom'; // Import routing components

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <BrowserRouter>
      {/* Optional: Basic Navigation Bar */}
      <nav style={{ padding: '10px 20px', backgroundColor: '#f0f0f0', marginBottom: '20px', textAlign: 'center' }}>
        <RouterLink to="/cam" style={{ marginRight: '15px', textDecoration: 'none', color: 'blue' }}>
          Home
        </RouterLink>
        <RouterLink to="/cam/test-subpage" style={{ textDecoration: 'none', color: 'blue' }}>
          Test Subpage
        </RouterLink>
        {/* You can add more links here for other routes */}
      </nav>

      {/* Define your application's routes */}
      <Routes>
        <Route path="/cam" element={<App />} /> {/* Main App component at the root path */}
        <Route path="/cam/test-subpage" element={<TestSubpage />} /> {/* Route for the test subpage */}
        {/* Add more <Route> components here for other pages */}
        {/* Example for a non-existent page (optional 404 handling in React):
        <Route path="*" element={<div><h2>404 - Page Not Found (Client Side)</h2><RouterLink to="/">Go Home</RouterLink></div>} />
        */}
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);

reportWebVitals();