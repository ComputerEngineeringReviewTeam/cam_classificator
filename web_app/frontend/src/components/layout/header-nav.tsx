import React from 'react';
import NavItem from './nav-item';

const HeaderNav: React.FC = () => {
  return (
    <div className="bg-[#282c34] p-5 text-white text-center flex-row md:flex items-center">
      <h1 className="m-0 text-3xl font-bold">CAM Classificator</h1>

      <nav className="flex-grow">
        <ul className="list-none p-0 m-0 flex justify-center md:justify-end">
          <NavItem name="Home" href="/cam" />
          <NavItem name="Classificator" href="/cam/classificator" />
          <NavItem name="Login" href="/cam/login" />
        </ul>
      </nav>
    </div>
  );
};

export default HeaderNav;