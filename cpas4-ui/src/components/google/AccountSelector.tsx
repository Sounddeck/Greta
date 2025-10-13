/**
 * Google Account Selector Component
 * 
 * Dropdown component for selecting which Google account to use for API calls.
 * Shows account email, name, and primary status.
 */

import React, { useEffect } from 'react';
import { ChevronDown, Mail, Star, Plus, Settings } from 'lucide-react';
import { useGoogleAccountStore } from '../../stores/googleAccountStore';

interface AccountSelectorProps {
  userId: string;
  onManageAccounts?: () => void;
  compact?: boolean;
}

export default function AccountSelector({
  userId,
  onManageAccounts,
  compact = false,
}: AccountSelectorProps) {
  const {
    accounts,
    selectedAccount,
    primaryAccount,
    loading,
    error,
    fetchAccounts,
    selectAccount,
    addAccount,
  } = useGoogleAccountStore();

  const [isOpen, setIsOpen] = React.useState(false);

  // Fetch accounts on mount
  useEffect(() => {
    if (userId) {
      fetchAccounts(userId);
    }
  }, [userId, fetchAccounts]);

  // Get selected account object
  const selectedAccountObj = accounts.find(
    (acc) => acc.account_email === selectedAccount
  );

  // Handle account selection
  const handleSelectAccount = (accountEmail: string) => {
    selectAccount(accountEmail);
    setIsOpen(false);
  };

  // Handle add account
  const handleAddAccount = async () => {
    setIsOpen(false);
    await addAccount(userId);
  };

  // Handle manage accounts
  const handleManageAccounts = () => {
    setIsOpen(false);
    onManageAccounts?.();
  };

  if (loading && accounts.length === 0) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 bg-gray-100 dark:bg-gray-800 rounded-lg">
        <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <span className="text-sm text-gray-600 dark:text-gray-400">
          Loading accounts...
        </span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 bg-red-50 dark:bg-red-900/20 rounded-lg">
        <span className="text-sm text-red-600 dark:text-red-400">{error}</span>
      </div>
    );
  }

  if (accounts.length === 0) {
    return (
      <button
        onClick={handleAddAccount}
        className="flex items-center gap-2 px-3 py-2 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-colors"
      >
        <Plus className="w-4 h-4" />
        <span className="text-sm font-medium">Add Google Account</span>
      </button>
    );
  }

  return (
    <div className="relative">
      {/* Selected Account Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 px-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors ${
          compact ? 'text-sm' : ''
        }`}
      >
        <Mail className="w-4 h-4 text-gray-500 dark:text-gray-400" />
        
        {selectedAccountObj ? (
          <div className="flex items-center gap-2">
            <div className="flex flex-col items-start">
              <div className="flex items-center gap-1">
                <span className="font-medium text-gray-900 dark:text-gray-100">
                  {selectedAccountObj.account_name || selectedAccountObj.account_email}
                </span>
                {selectedAccountObj.is_primary && (
                  <Star className="w-3 h-3 text-yellow-500 fill-yellow-500" />
                )}
              </div>
              {!compact && selectedAccountObj.account_name && (
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {selectedAccountObj.account_email}
                </span>
              )}
            </div>
          </div>
        ) : (
          <span className="text-gray-500 dark:text-gray-400">Select account</span>
        )}
        
        <ChevronDown
          className={`w-4 h-4 text-gray-500 dark:text-gray-400 transition-transform ${
            isOpen ? 'rotate-180' : ''
          }`}
        />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />

          {/* Menu */}
          <div className="absolute top-full left-0 mt-2 w-80 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-20 overflow-hidden">
            {/* Accounts List */}
            <div className="max-h-64 overflow-y-auto">
              {accounts.map((account) => (
                <button
                  key={account.account_email}
                  onClick={() => handleSelectAccount(account.account_email)}
                  className={`w-full flex items-center gap-3 px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors ${
                    account.account_email === selectedAccount
                      ? 'bg-blue-50 dark:bg-blue-900/20'
                      : ''
                  }`}
                >
                  <div className="flex-shrink-0 w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-semibold text-sm">
                    {account.account_name?.[0]?.toUpperCase() ||
                      account.account_email[0].toUpperCase()}
                  </div>

                  <div className="flex-1 flex flex-col items-start">
                    <div className="flex items-center gap-1">
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {account.account_name || account.account_email}
                      </span>
                      {account.is_primary && (
                        <Star className="w-3 h-3 text-yellow-500 fill-yellow-500" />
                      )}
                    </div>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {account.account_email}
                    </span>
                  </div>

                  {account.account_email === selectedAccount && (
                    <div className="w-2 h-2 bg-blue-500 rounded-full" />
                  )}
                </button>
              ))}
            </div>

            {/* Divider */}
            <div className="border-t border-gray-200 dark:border-gray-700" />

            {/* Actions */}
            <div className="p-2">
              <button
                onClick={handleAddAccount}
                className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-750 rounded-md transition-colors"
              >
                <Plus className="w-4 h-4" />
                <span>Add another account</span>
              </button>

              {onManageAccounts && (
                <button
                  onClick={handleManageAccounts}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-750 rounded-md transition-colors"
                >
                  <Settings className="w-4 h-4" />
                  <span>Manage accounts</span>
                </button>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

