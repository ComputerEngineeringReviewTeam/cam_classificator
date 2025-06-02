import React from 'react';
import { User as AuthUser } from '../../contexts/auth-context';

interface UserTableRowProps {
  user: AuthUser;
  currentUser: AuthUser | null;
  onDeleteUser: (userId: number, username: string) => void;
}

const UserTableRow: React.FC<UserTableRowProps> = ({ user, currentUser, onDeleteUser }) => {
  return (
    <tr key={user.id} className="hover:bg-gray-50">
      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{user.id}</td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{user.username}</td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
          user.is_admin ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
        }`}>
          {user.is_admin ? 'Admin' : 'User'}
        </span>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
        <button
          onClick={() => onDeleteUser(user.id, user.username)}
          disabled={currentUser?.id === user.id}
          className="text-red-600 hover:text-red-900 disabled:text-gray-400 disabled:cursor-not-allowed"
        >
          Delete
        </button>
      </td>
    </tr>
  );
};

export default UserTableRow;