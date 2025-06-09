import React from 'react';
import { Link } from 'react-router-dom';

const NotFoundPage: React.FC = () => {
  return (
    <div>
      <h1 className="text-red-700 text-5xl">404 - Page Not Found</h1>
      <p>Sorry, the page you are looking for does not exist.</p>
      <Link to="/cam" className="underline text-blue-500 hover:text-blue-800">Go to Home Page</Link>
    </div>
  );
};

export default NotFoundPage;