import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5000/cam/api', // Your Flask backend URL
  withCredentials: true, // Important for sending/receiving session cookies
});

export default api;
