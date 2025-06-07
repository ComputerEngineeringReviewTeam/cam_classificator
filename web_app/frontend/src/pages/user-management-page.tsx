import React, { useState, useEffect } from 'react';
import api from '../services/api';
import { useAuth, User as AuthUser } from '../models/auth-context';
import CreateUserForm from '../components/admin/create-user-form';
import UsersTable from '../components/admin/users-table';

const AdminUserManagementPage: React.FC = () => {
  const { currentUser } = useAuth();
  const [users, setUsers] = useState<AuthUser[]>([]);
  const [isLoadingUsers, setIsLoadingUsers] = useState(false);
  const [fetchUsersError, setFetchUsersError] = useState<string | null>(null);

  const fetchUsers = async () => {
    setIsLoadingUsers(true);
    setFetchUsersError(null);
    try {
      const response = await api.get<AuthUser[]>('/admin/users');
      setUsers(response.data);
    } catch (err: any) {
      setFetchUsersError(err.response?.data?.message || 'Failed to fetch users.');
      console.error(err);
    } finally {
      setIsLoadingUsers(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const handleUserCreated = (newUser: AuthUser) => {
    setUsers(prevUsers => [...prevUsers, newUser]);
  };

  const handleDeleteUser = async (userId: number, username: string) => {
    if (currentUser && currentUser.id === userId) {
      alert("You cannot delete your own account from this interface.");
      return;
    }

    if (window.confirm(`Are you sure you want to delete user "${username}" (ID: ${userId})? This action cannot be undone.`)) {
      try {
        await api.delete(`/admin/users/${userId}`);
        setUsers(prevUsers => prevUsers.filter(user => user.id !== userId));
        alert(`User "${username}" deleted successfully.`);
      } catch (err: any) {
        alert(err.response?.data?.message || `Failed to delete user "${username}".`);
        console.error(err);
        await fetchUsers();
      }
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h2 className="text-3xl font-bold mb-6 text-gray-800">User Management</h2>

      <CreateUserForm onUserCreated={handleUserCreated} />

      <h3 className="text-xl font-semibold mt-8 mb-4 text-gray-700">Existing Users</h3>
      <UsersTable
        users={users}
        currentUser={currentUser}
        onDeleteUser={handleDeleteUser}
        isLoading={isLoadingUsers}
        error={fetchUsersError}
      />
    </div>
  );
};

export default AdminUserManagementPage;