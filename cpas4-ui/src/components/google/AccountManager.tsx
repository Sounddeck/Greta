/**
 * Google Account Manager Component
 * 
 * Full-featured account management interface for Google Workspace accounts.
 * Allows users to view, add, remove, and manage multiple Google accounts.
 */

import React, { useEffect, useState } from 'react';
import {
  Mail,
  Star,
  Plus,
  Trash2,
  Calendar,
  Clock,
  Shield,
  AlertCircle,
  CheckCircle,
} from 'lucide-react';
import { useGoogleAccountStore } from '../../stores/googleAccountStore';
import { toast } from 'sonner';

interface AccountManagerProps {
  userId: string;
}

export default function AccountManager({ userId }: AccountManagerProps) {
  const {
    accounts,
    primaryAccount,
    loading,
    error,
    fetchAccounts,
    addAccount,
    removeAccount,
    setPrimaryAccount,
    clearError,
  } = useGoogleAccountStore();

  const [confirmRemove, setConfirmRemove] = useState<string | null>(null);

  // Fetch accounts on mount
  useEffect(() => {
    if (userId) {
      fetchAccounts(userId);
    }
  }, [userId, fetchAccounts]);

  // Handle add account
  const handleAddAccount = async () => {
    try {
      await addAccount(userId);
      // Note: Success toast will be shown after OAuth flow completes
      // The addAccount function opens OAuth window and polls for completion
    } catch (err) {
      toast.error('Failed to add account');
    }
  };

  // Handle remove account
  const handleRemoveAccount = async (accountEmail: string) => {
    try {
      await removeAccount(userId, accountEmail);
      toast.success('Account removed successfully');
      setConfirmRemove(null);
    } catch (err) {
      toast.error('Failed to remove account');
    }
  };

  // Handle set primary
  const handleSetPrimary = async (accountEmail: string) => {
    try {
      await setPrimaryAccount(userId, accountEmail);
      toast.success('Primary account updated');
    } catch (err) {
      toast.error('Failed to set primary account');
    }
  };

  // Format date
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  // Check if token is expired
  const isExpired = (expiresAt: string) => {
    return new Date(expiresAt) < new Date();
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
          Google Accounts
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Manage your connected Google Workspace accounts for Gmail, Calendar, and Drive.
        </p>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="mb-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
          </div>
          <button
            onClick={clearError}
            className="text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300"
          >
            ×
          </button>
        </div>
      )}

      {/* Add Account Button */}
      <div className="mb-6">
        <button
          onClick={handleAddAccount}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Plus className="w-4 h-4" />
          <span>Add Google Account</span>
        </button>
      </div>

      {/* Accounts List */}
      {loading && accounts.length === 0 ? (
        <div className="flex items-center justify-center py-12">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : accounts.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 dark:bg-gray-800 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-700">
          <Mail className="w-12 h-12 text-gray-400 dark:text-gray-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            No accounts connected
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Add a Google account to access Gmail, Calendar, and Drive.
          </p>
          <button
            onClick={handleAddAccount}
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span>Add Your First Account</span>
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {accounts.map((account) => {
            const expired = isExpired(account.expires_at);

            return (
              <div
                key={account.account_email}
                className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6"
              >
                {/* Account Header */}
                <div className="flex items-start gap-4 mb-4">
                  {/* Avatar */}
                  <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-semibold text-lg">
                    {account.account_name?.[0]?.toUpperCase() ||
                      account.account_email[0].toUpperCase()}
                  </div>

                  {/* Account Info */}
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                        {account.account_name || account.account_email}
                      </h3>
                      {account.is_primary && (
                        <span className="flex items-center gap-1 px-2 py-0.5 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 text-xs font-medium rounded-full">
                          <Star className="w-3 h-3 fill-current" />
                          Primary
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {account.account_email}
                    </p>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2">
                    {!account.is_primary && (
                      <button
                        onClick={() => handleSetPrimary(account.account_email)}
                        disabled={loading}
                        className="px-3 py-1.5 text-sm text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-750 disabled:opacity-50 transition-colors"
                      >
                        Set as Primary
                      </button>
                    )}

                    <button
                      onClick={() => setConfirmRemove(account.account_email)}
                      disabled={loading}
                      className="p-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-md disabled:opacity-50 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Account Details */}
                <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                  {/* Token Status */}
                  <div className="flex items-start gap-2">
                    {expired ? (
                      <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
                    ) : (
                      <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
                    )}
                    <div>
                      <p className="text-xs font-medium text-gray-700 dark:text-gray-300">
                        Token Status
                      </p>
                      <p
                        className={`text-xs ${
                          expired
                            ? 'text-red-600 dark:text-red-400'
                            : 'text-green-600 dark:text-green-400'
                        }`}
                      >
                        {expired ? 'Expired' : 'Active'}
                      </p>
                    </div>
                  </div>

                  {/* Expires At */}
                  <div className="flex items-start gap-2">
                    <Clock className="w-4 h-4 text-gray-500 dark:text-gray-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-xs font-medium text-gray-700 dark:text-gray-300">
                        Expires
                      </p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {formatDate(account.expires_at)}
                      </p>
                    </div>
                  </div>

                  {/* Created At */}
                  <div className="flex items-start gap-2">
                    <Calendar className="w-4 h-4 text-gray-500 dark:text-gray-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-xs font-medium text-gray-700 dark:text-gray-300">
                        Connected
                      </p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {formatDate(account.created_at)}
                      </p>
                    </div>
                  </div>

                  {/* Scopes */}
                  <div className="flex items-start gap-2">
                    <Shield className="w-4 h-4 text-gray-500 dark:text-gray-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-xs font-medium text-gray-700 dark:text-gray-300">
                        Permissions
                      </p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {account.scopes.length} scope{account.scopes.length !== 1 ? 's' : ''}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Confirm Remove Dialog */}
                {confirmRemove === account.account_email && (
                  <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                    <p className="text-sm text-red-900 dark:text-red-100 mb-3">
                      Are you sure you want to remove this account? This will revoke access
                      to Gmail, Calendar, and Drive for{' '}
                      <strong>{account.account_email}</strong>.
                    </p>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleRemoveAccount(account.account_email)}
                        disabled={loading}
                        className="px-3 py-1.5 bg-red-600 text-white text-sm rounded-md hover:bg-red-700 disabled:opacity-50 transition-colors"
                      >
                        Remove Account
                      </button>
                      <button
                        onClick={() => setConfirmRemove(null)}
                        disabled={loading}
                        className="px-3 py-1.5 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-sm rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50 transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Info Box */}
      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
        <h4 className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-2">
          About Multiple Accounts
        </h4>
        <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
          <li>• Connect multiple Google accounts (personal, work, etc.)</li>
          <li>• Set a primary account for default operations</li>
          <li>• Switch between accounts when using Gmail, Calendar, or Drive</li>
          <li>• Each account maintains separate permissions and data</li>
        </ul>
      </div>
    </div>
  );
}

