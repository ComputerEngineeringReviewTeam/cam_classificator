import React from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../../contexts/auth-context';

interface ProtectedRouteProps {
  adminOnly?: boolean;
  children?: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ adminOnly = false, children }) => {
  const { currentUser, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return <div>Waiting for authentication status...</div>;
  }

  if (!currentUser) {
    return <Navigate to="/cam" state={{ from: location }} replace />;
  }

  if (adminOnly && !currentUser.is_admin) {
    return <Navigate to="/cam/forbidden" replace />;
  }

  return children ? <>{children}</> : <Outlet />;
};

export default ProtectedRoute;