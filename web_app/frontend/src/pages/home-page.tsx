import React from 'react';
import { useAuth } from '../models/auth-context';

const HomePage: React.FC = () => {
  const { currentUser } = useAuth();

  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-200px)] py-10 px-4 text-center">
      <div className="max-w-2xl w-full bg-white p-8 md:p-12 rounded-xl shadow-2xl shadow-blue-300">
        <h2 className="text-4xl md:text-5xl font-bold text-gray-800 mb-6">
          Welcome to CAM Classificator!
        </h2>

        {currentUser ? (
          <p className="text-lg text-gray-700 mb-8">
            Hello, <span className="font-semibold">{currentUser.username}</span>! Go and start using the classificator now.
          </p>
        ) : (
          <p className="text-lg text-gray-700 mb-8">
            This is the main landing page of our awesome application. Please log in to access all features.
          </p>
        )}

        <div className="space-y-4">
          <p className="text-md text-gray-600">
            Navigate using the links above to explore different sections of the application.
          </p>
          {currentUser ? (
            <a
              href="/cam/classificator"
              className="inline-block bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-8 rounded-lg shadow-md hover:shadow-lg transition-transform duration-150 ease-in-out transform hover:-translate-y-0.5"
            >
              Go to the classificator
            </a>
          ) : (
            <a
              href="/cam/login"
              className="inline-block bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-8 rounded-lg shadow-md hover:shadow-lg transition-transform duration-150 ease-in-out transform hover:-translate-y-0.5"
            >
              Go to Login
            </a>
          )}
        </div>
      </div>
    </div>
  );
};

export default HomePage;