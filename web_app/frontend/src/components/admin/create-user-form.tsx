import React, { useState, FormEvent } from 'react';
import api from '../../services/api';
import { User as AuthUser } from '../../models/auth-context';

interface CreateUserFormProps {
  onUserCreated: (newUser: AuthUser) => void;
}

const CreateUserForm: React.FC<CreateUserFormProps> = ({ onUserCreated }) => {
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [isCreatingUser, setIsCreatingUser] = useState(false);
  const [createUserError, setCreateUserError] = useState<string | null>(null);

  const handleCreateUser = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsCreatingUser(true);
    setCreateUserError(null);
    try {
      const response = await api.post<{ message: string; user: AuthUser }>('/admin/users', {
        username: newUsername,
        password: newPassword,
        is_admin: false,
      });
      onUserCreated(response.data.user);
      setNewUsername('');
      setNewPassword('');
      alert(response.data.message || 'User created successfully!');
    } catch (err: any) {
      setCreateUserError(err.response?.data?.message || 'Failed to create user.');
      console.error(err);
    } finally {
      setIsCreatingUser(false);
    }
  };

  return (
    <div className="mb-8 p-6 bg-white rounded-lg shadow-md">
      <h3 className="text-xl font-semibold mb-4 text-gray-700">Create New User</h3>
      <form onSubmit={handleCreateUser}>
        {createUserError && <p className="text-red-500 text-sm mb-3">{createUserError}</p>}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label htmlFor="newUsername" className="block text-sm font-medium text-gray-700 mb-1">Username</label>
            <input
              type="text"
              id="newUsername"
              value={newUsername}
              onChange={(e) => setNewUsername(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
            />
          </div>
          <div>
            <label htmlFor="newPassword" className="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <input
              type="password"
              id="newPassword"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              required
              minLength={6}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
            />
          </div>
        </div>
        <button
          type="submit"
          disabled={isCreatingUser}
          className="w-full md:w-auto px-4 py-2 bg-green-600 text-white font-semibold rounded-md shadow-sm hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50"
        >
          {isCreatingUser ? 'Creating...' : 'Create User'}
        </button>
      </form>
    </div>
  );
};

export default CreateUserForm;