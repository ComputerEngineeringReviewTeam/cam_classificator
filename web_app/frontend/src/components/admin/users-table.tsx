import React from 'react';
import { User as AuthUser } from '../../models/auth-context';
import UserTableRow from './user-table-row';

interface UsersTableProps {
  users: AuthUser[];
  currentUser: AuthUser | null;
  onDeleteUser: (userId: number, username: string) => void;
  isLoading: boolean;
  error: string | null;
}

const UsersTable: React.FC<UsersTableProps> = ({ users, currentUser, onDeleteUser, isLoading, error }) => {
  if (isLoading) {
    return <p className="text-blue-500">Loading users...</p>;
  }

  if (error) {
    return <p className="text-red-500 bg-red-100 p-3 rounded-md">{error}</p>;
  }

  return (
    <div className="overflow-x-auto bg-white rounded-lg shadow">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Username</th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Admin Status</th>
            <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {users.length > 0 ? (
            users.map((user) => (
              <UserTableRow
                key={user.id}
                user={user}
                currentUser={currentUser}
                onDeleteUser={onDeleteUser}
              />
            ))
          ) : (
            <tr>
              <td colSpan={4} className="px-6 py-4 text-center text-sm text-gray-500">
                No users found.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
};

export default UsersTable;