/**
 * Google Account Settings Page
 * 
 * Settings page for managing Google Workspace accounts.
 */

import React from 'react';
import AccountManager from '../components/google/AccountManager';

export default function GoogleAccountSettings() {
  // TODO: Get actual user ID from auth context
  const userId = 'test_user';

  return (
    <div className="h-full overflow-y-auto">
      <AccountManager userId={userId} />
    </div>
  );
}

