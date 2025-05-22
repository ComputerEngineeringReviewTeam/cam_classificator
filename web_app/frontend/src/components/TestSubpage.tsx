// frontend/src/components/TestSubpage.tsx
import React from 'react';
import { Link } from 'react-router-dom';

const TestSubpage: React.FC = () => {
  return (
    <div style={{ padding: '20px', textAlign: 'center' }}>
      <h2>This is the Test Subpage!</h2>
      <p>If you can see this, client-side routing to this subpage is working correctly.</p>
      <p>
        When you refresh this page, or navigate directly to <code>/test-subpage</code>,
        Flask should serve the main <code>index.html</code>, and then React Router
        takes over to display this component.
      </p>
      <Link to="/">Go back to Home</Link>
    </div>
  );
};

export default TestSubpage;