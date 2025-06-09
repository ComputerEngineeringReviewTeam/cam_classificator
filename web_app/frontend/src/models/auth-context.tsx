import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';
import api from '../services/api';
import { useNavigate } from 'react-router-dom';

export interface User {
  id: number;
  username: string;
  is_admin: boolean;
}

interface AuthContextType {
  currentUser: User | null;
  login: (usernameOrEmail: string, password: string) => Promise<User>;
  logout: () => Promise<void>;
  isLoading: boolean;
  setCurrentUser: React.Dispatch<React.SetStateAction<User | null>>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const checkLoggedInStatus = async () => {
      setIsLoading(true);
      try {
        const response = await api.get<{ logged_in: boolean; user?: User }>('/auth/status');
        if (response.data.logged_in && response.data.user) {
          setCurrentUser(response.data.user);
        } else {
          setCurrentUser(null);
        }
      } catch (error) {
        setCurrentUser(null);
      } finally {
        setIsLoading(false);
      }
    };
    checkLoggedInStatus();
  }, []);

  const login = async (username: string, password_val: string): Promise<User> => {
    try {
      const response = await api.post<{ message: string; user: User }>('/auth/login', {
        username: username,
        password: password_val,
      });
      setCurrentUser(response.data.user);
      return response.data.user;
    } catch (error) {
      throw error;
    }
  };

  const logout = async () => {
    try {
      await api.post('/auth/logout');
      setCurrentUser(null);
      navigate('/cam');
    } catch (error) {
      console.error('Logout failed:', error);
      setCurrentUser(null);

      throw error;
    }
  };

  return (
    <AuthContext.Provider value={{ currentUser, login, logout, isLoading, setCurrentUser }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};