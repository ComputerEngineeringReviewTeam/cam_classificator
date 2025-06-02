import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/layout/layout';
import HomePage from './pages/home-page';
import NotFoundPage from './pages/not-found-page';
import ForbiddenPage from './pages/forbidden';
import LoginPage from './pages/login-page';
import ProtectedRoute from './components/auth/protected-route';
import './App.css';

// TEMPORARY
const AdminUserManagementPage: React.FC = () => <div><h2>Admin User Management (Admin Only)</h2><p>Here admins can manage users.</p></div>;
const ClassificatorPage: React.FC = () => <div><h2>Classificator (Protected)</h2><p>Only logged-in users can use the classificator.</p></div>;

// add forbidden page
const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/cam" element={<Layout />}>
        {/*--- PUBLIC ---*/}
        <Route index element={<HomePage />} />
        <Route path="login" element={<LoginPage />} />

        {/*--- LOGIN REQUIRED ---*/}
        <Route element={<ProtectedRoute />}>
          <Route path="classificator" element={<ClassificatorPage />} />
        </Route>


        {/*--- ADMIN ONLY ---*/}
        <Route element={<ProtectedRoute adminOnly={true}/>}>
          <Route path="admin/users" element={<AdminUserManagementPage />} />
        </Route>

        {/*--- OTHER ---*/}
        <Route path="forbidden" element={<ForbiddenPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
};

export default App;