import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_SERVER_ADDR + '/cam/api',
  withCredentials: true,
});

export default api;
