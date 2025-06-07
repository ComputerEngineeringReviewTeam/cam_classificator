import React from 'react';
import { Outlet } from 'react-router-dom';
import HeaderNav from './header-nav';

const Layout: React.FC = () => {
  return (
    <div className="flex flex-col min-h-screen">
      <HeaderNav />
      <main className="flex-grow p-5 bg-gradient-to-tl from-sky-50 to-sky-300">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;