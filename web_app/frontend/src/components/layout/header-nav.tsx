import React from 'react';
import NavItem from './nav-item';
import { useAuth} from "../../contexts/auth-context";
import { useNavigate } from 'react-router-dom';

const HeaderNav: React.FC = () => {
  const { currentUser, logout, isLoading } = useAuth(); // Get user and logout function
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/cam/login');
    } catch (error) {
      console.error("Failed to logout:", error);
      // Handle error display if needed
    }
  };

  return (
    <div className="bg-[#282c34] p-5 text-white text-center flex-row md:flex items-center">
      <h1 className="m-0 text-3xl font-bold">CAM Classificator</h1>

      <nav className="flex-grow">
        <ul className="list-none p-0 m-0 flex justify-center md:justify-end">
          <NavItem name="Home" href="/cam" />
          {currentUser && <NavItem name="Classificator" href="/cam/classificator" />}

          {isLoading ? (
            <li className="mx-4 text-xl font-bold">Loading...</li>
          ) : currentUser ? (
            <>
              {currentUser.is_admin && (
                <NavItem name="Manage Users" href="/cam/admin/users" />
              )}
              <li className="mx-4 text-xl">
                <span className="mr-2">Hi, {currentUser.username}!</span>
                <button
                  onClick={handleLogout}
                  className="bg-red-500 hover:bg-red-700 text-white font-bold py-1 px-3 rounded text-sm"
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