/**
 * MCP Store Initializer
 * Automatically loads MCP servers and tools from MongoDB on app startup
 * 
 * This component fixes the issue where MCP servers disappear after page refresh
 * by ensuring data is always loaded from the database, not just localStorage.
 */

import { useEffect, useState } from 'react';
import { useMCPStore } from '../stores/mcpStore';
import { toast } from 'sonner';

export function MCPStoreInitializer() {
  const [initialized, setInitialized] = useState(false);
  const initialize = useMCPStore((state) => state.initialize);

  useEffect(() => {
    const initializeStore = async () => {
      if (initialized) return;

      try {
        await initialize();
        setInitialized(true);
      } catch (error) {
        // Silently handle initialization errors
        // Backend might not be available
      }
    };

    initializeStore();
  }, [initialize, initialized]);

  // This component doesn't render anything
  return null;
}

export default MCPStoreInitializer;

