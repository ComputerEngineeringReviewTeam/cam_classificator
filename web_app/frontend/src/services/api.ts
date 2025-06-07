import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5000/cam/api',
  withCredentials: true,
});

export default api;
