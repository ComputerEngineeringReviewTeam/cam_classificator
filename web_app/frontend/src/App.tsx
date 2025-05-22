// frontend/src/App.tsx
import React, { useState, useEffect } from 'react';
import logo from './logo.svg';
import './App.css';

// Define types for your API responses for better type safety
interface HelloApiResponse {
  message: string;
}

interface DataApiResponse {
  items: string[]; // Assuming items are strings, adjust if they are objects
}

// Example of an item type if your data items are objects
// interface MyItem {
//   id: number;
//   name: string;
// }
// interface DataApiResponse {
//   items: MyItem[];
// }

function App() {
  const [message, setMessage] = useState<string>(''); // Explicitly type the state
  const [items, setItems] = useState<string[]>([]);    // Explicitly type the state

  useEffect(() => {
    // Fetch a message from /api/hello
    fetch('/cam/api/hello')
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json() as Promise<HelloApiResponse>; // Type assertion for the response
      })
      .then(data => {
        setMessage(data.message);
      })
      .catch(error => console.error('Error fetching hello:', error));

    // Fetch items from /api/data
    fetch('/cam/api/data')
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json() as Promise<DataApiResponse>; // Type assertion
      })
      .then(data => {
        setItems(data.items || []); // Ensure items is an array
      })
      .catch(error => console.error('Error fetching data:', error));
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload. (Now with TypeScript!)
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
        <p>Message from Flask: {message}</p>
        <div>
          <h3>Items from Flask API:</h3>
          <ul>
            {items.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </div>
      </header>
    </div>
  );
}

export default App;