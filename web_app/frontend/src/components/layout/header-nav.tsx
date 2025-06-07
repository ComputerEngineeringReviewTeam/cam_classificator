import React from 'react';
import NavItem from './nav-item';
import { useAuth} from "../../models/auth-context";
import { useNavigate } from 'react-router-dom';

const HeaderNav: React.FC = () => {
  const { currentUser, logout, isLoading } = useAuth(); // Get user and logout function
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/cam');
    } catch (error) {
      console.error("Failed to logout:", error);
      navigate('/cam');
    }
  };

  return (
    <div className="bg-[#282c34] p-5 text-white text-center flex-row md:flex items-center">
      <h1 className="m-0 text-3xl font-bold">CAM Classificator</h1>

      <nav className="flex-grow">
        <ul className="list-none p-0 m-0 flex-row justify-center md:justify-end md:flex">
          <NavItem name="Home" href="/cam" />
          {currentUser && <NavItem name="Classificator" href="/cam/classificator" />}

          {isLoading ? (
            <li className="mx-4 py-2 px-4 text-xl font-bold">Loading...</li>
          ) : currentUser ? (
            <>
              {currentUser.is_admin && (
                <NavItem name="Manage Users" href="/cam/admin/users" />
              )}
              <li className="mx-4 md:flex items-center">
                <span className="mr-3 py-2 text-xl font-bold">Hi, {currentUser.username}!</span>
                <button
                  onClick={handleLogout}
                  className="bg-red-500 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded text-lg transition-transform duration-200 ease-in-out hover:scale-105 active:scale-105"
                >
                  Logout
                </button>
              </li>
            </>
          ) : (
            <NavItem name="Login" href="/cam/login" />
          )}
        </ul>
      </nav>
    </div>
  );
};

export default HeaderNav;