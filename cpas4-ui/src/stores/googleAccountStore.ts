/**
 * Google Account Store
 * 
 * Manages Google Workspace account state including:
 * - List of connected accounts
 * - Selected account for API calls
 * - Primary account designation
 * - Account management operations
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// ============================================================================
// Types
// ============================================================================

export interface GoogleAccount {
  account_email: string;
  account_name: string | null;
  picture_url?: string | null;
  is_primary: boolean;
  token_id: string;
  expires_at: string;
  scopes: string[];
  created_at: string;
}

export interface GoogleAccountListResponse {
  accounts: GoogleAccount[];
  total_accounts: number;
  primary_account: string | null;
}

// ============================================================================
// Store Interface
// ============================================================================

interface GoogleAccountState {
  // State
  accounts: GoogleAccount[];
  selectedAccount: string | null; // account_email
  primaryAccount: string | null; // account_email
  loading: boolean;
  error: string | null;
  
  // Actions
  fetchAccounts: (userId: string) => Promise<void>;
  selectAccount: (accountEmail: string) => void;
  addAccount: (userId: string) => Promise<void>;
  removeAccount: (userId: string, accountEmail: string) => Promise<void>;
  setPrimaryAccount: (userId: string, accountEmail: string) => Promise<void>;
  clearError: () => void;
  reset: () => void;
}

// ============================================================================
// Initial State
// ============================================================================

const initialState = {
  accounts: [],
  selectedAccount: null,
  primaryAccount: null,
  loading: false,
  error: null,
};

// ============================================================================
// Store
// ============================================================================

export const useGoogleAccountStore = create<GoogleAccountState>()(
  persist(
    (set, get) => ({
      ...initialState,
      
      // ======================================================================
      // Fetch Accounts
      // ======================================================================
      
      fetchAccounts: async (userId: string) => {
        set({ loading: true, error: null });
        
        try {
          const response = await fetch(
            `http://localhost:8000/api/v1/oauth/accounts/google?user_id=${userId}`
          );
          
          if (!response.ok) {
            throw new Error(`Failed to fetch accounts: ${response.statusText}`);
          }
          
          const data: GoogleAccountListResponse = await response.json();
          
          set({
            accounts: data.accounts,
            primaryAccount: data.primary_account,
            // If no account selected, use primary or first account
            selectedAccount: get().selectedAccount || data.primary_account || (data.accounts[0]?.account_email || null),
            loading: false,
          });
        } catch (error) {
          console.error('Error fetching Google accounts:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to fetch accounts',
            loading: false,
          });
        }
      },
      
      // ======================================================================
      // Select Account
      // ======================================================================
      
      selectAccount: (accountEmail: string) => {
        const { accounts } = get();
        const account = accounts.find(a => a.account_email === accountEmail);
        
        if (account) {
          set({ selectedAccount: accountEmail });
        } else {
          console.warn(`Account ${accountEmail} not found`);
        }
      },
      
      // ======================================================================
      // Add Account
      // ======================================================================
      
      addAccount: async (userId: string) => {
        set({ loading: true, error: null });
        
        try {
          // Initiate OAuth flow
          const response = await fetch(
            `http://localhost:8000/api/v1/oauth/authorize?user_id=${userId}&provider=google&scopes=gmail.readonly,calendar,drive.file`
          );
          
          if (!response.ok) {
            throw new Error(`Failed to initiate OAuth: ${response.statusText}`);
          }
          
          const data = await response.json();
          
          // Open authorization URL in new window
          const authWindow = window.open(
            data.authorization_url,
            'Google OAuth',
            'width=600,height=700'
          );
          
          // Poll for completion (in real implementation, use callback)
          const pollInterval = setInterval(async () => {
            if (authWindow?.closed) {
              clearInterval(pollInterval);
              // Refresh accounts list
              await get().fetchAccounts(userId);
              set({ loading: false });
            }
          }, 1000);
          
        } catch (error) {
          console.error('Error adding Google account:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to add account',
            loading: false,
          });
        }
      },
      
      // ======================================================================
      // Remove Account
      // ======================================================================
      
      removeAccount: async (userId: string, accountEmail: string) => {
        set({ loading: true, error: null });
        
        try {
          const response = await fetch(
            `http://localhost:8000/api/v1/oauth/accounts/google/${encodeURIComponent(accountEmail)}?user_id=${userId}`,
            { method: 'DELETE' }
          );
          
          if (!response.ok) {
            throw new Error(`Failed to remove account: ${response.statusText}`);
          }
          
          // Refresh accounts list
          await get().fetchAccounts(userId);
          
          // If removed account was selected, clear selection
          if (get().selectedAccount === accountEmail) {
            const { accounts, primaryAccount } = get();
            set({
              selectedAccount: primaryAccount || (accounts[0]?.account_email || null),
            });
          }
          
          set({ loading: false });
        } catch (error) {
          console.error('Error removing Google account:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to remove account',
            loading: false,
          });
        }
      },
      
      // ======================================================================
      // Set Primary Account
      // ======================================================================
      
      setPrimaryAccount: async (userId: string, accountEmail: string) => {
        set({ loading: true, error: null });
        
        try {
          const response = await fetch(
            'http://localhost:8000/api/v1/oauth/accounts/google/primary',
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                user_id: userId,
                account_email: accountEmail,
              }),
            }
          );
          
          if (!response.ok) {
            throw new Error(`Failed to set primary account: ${response.statusText}`);
          }
          
          // Refresh accounts list
          await get().fetchAccounts(userId);
          set({ loading: false });
        } catch (error) {
          console.error('Error setting primary account:', error);
          set({
            error: error instanceof Error ? error.message : 'Failed to set primary account',
            loading: false,
          });
        }
      },
      
      // ======================================================================
      // Utility Actions
      // ======================================================================
      
      clearError: () => set({ error: null }),
      
      reset: () => set(initialState),
    }),
    {
      name: 'google-account-storage',
      partialize: (state) => ({
        selectedAccount: state.selectedAccount,
        // Don't persist accounts, primaryAccount - fetch fresh on load
      }),
    }
  )
);

