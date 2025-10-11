import { useEffect, useState } from 'react';
import { Webhook, Plus, Trash2, TestTube, Loader2, Zap, Package } from 'lucide-react';
import { Button } from '../ui/button';
import { useHookStore } from '../../stores/hookStore';
import { HookSkeletonList, PluginSkeleton, EventTypeSkeleton } from '../skeletons/HookSkeleton';

export function HookManager() {
  // Use Zustand store
  const {
    hooks,
    plugins,
    eventTypes,
    isLoading,
    isExecuting,
    fetchHooks,
    fetchPlugins,
    fetchEventTypes,
    registerHook,
    deleteHook,
    testHook,
  } = useHookStore();

  // Local state for form
  const [hookName, setHookName] = useState('');
  const [selectedEventType, setSelectedEventType] = useState('');
  const [selectedPlugin, setSelectedPlugin] = useState('');
  const [priority, setPriority] = useState(100);

  // Load data on mount
  useEffect(() => {
    fetchHooks();
    fetchPlugins();
    fetchEventTypes();
  }, [fetchHooks, fetchPlugins, fetchEventTypes]);

  const handleRegisterHook = async () => {
    if (!hookName || !selectedEventType || !selectedPlugin) return;

    await registerHook({
      name: hookName,
      eventType: selectedEventType,
      pluginId: selectedPlugin,
      isEnabled: true,
      priority,
    });

    // Reset form
    setHookName('');
    setSelectedEventType('');
    setSelectedPlugin('');
    setPriority(100);
  };

  const handleDeleteHook = async (id: string) => {
    if (confirm('Are you sure you want to delete this hook?')) {
      await deleteHook(id);
    }
  };

  const handleTestHook = async (id: string) => {
    await testHook(id, { test: true, timestamp: new Date().toISOString() });
  };

  return (
    <div className="space-y-6 p-6">
      <div>
        <h2 className="text-2xl font-bold mb-2">Hook Manager</h2>
        <p className="text-muted-foreground">
          Manage event hooks and plugin integrations
        </p>
      </div>

      {/* Registered Hooks - Show skeleton during initial load */}
      {isLoading && hooks.length === 0 ? (
        <div className="space-y-3">
          <h3 className="font-semibold">Registered Hooks</h3>
          <HookSkeletonList count={3} />
        </div>
      ) : hooks.length > 0 ? (
        <div className="space-y-3">
          <h3 className="font-semibold">Registered Hooks</h3>
          <div className="grid gap-2">
            {hooks.map((hook) => (
              <div key={hook.id} className="p-3 border border-border rounded-md">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <Webhook className="h-4 w-4 text-primary" />
                      <p className="font-medium">{hook.name}</p>
                      {hook.isEnabled && (
                        <span className="text-xs px-2 py-0.5 bg-green-500/20 text-green-600 rounded-full">
                          Enabled
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                      <span>Event: {hook.eventType}</span>
                      <span>Priority: {hook.priority}</span>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleTestHook(hook.id)}
                      disabled={isExecuting}
                    >
                      {isExecuting ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <TestTube className="h-4 w-4" />
                      )}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDeleteHook(hook.id)}
                      disabled={isLoading}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {/* Available Plugins - Show skeleton during initial load */}
      {isLoading && plugins.length === 0 ? (
        <div className="space-y-3">
          <h3 className="font-semibold">Available Plugins</h3>
          <div className="grid md:grid-cols-2 gap-2">
            {[1, 2, 3, 4].map((i) => (
              <PluginSkeleton key={i} />
            ))}
          </div>
        </div>
      ) : plugins.length > 0 ? (
        <div className="space-y-3">
          <h3 className="font-semibold">Available Plugins</h3>
          <div className="grid md:grid-cols-2 gap-2">
            {plugins.map((plugin) => (
              <div key={plugin.id} className="p-3 bg-muted rounded-md">
                <div className="flex items-center gap-2">
                  <Package className="h-4 w-4 text-primary" />
                  <p className="font-medium">{plugin.name}</p>
                  <span className="text-xs text-muted-foreground">v{plugin.version}</span>
                </div>
                {plugin.description && (
                  <p className="text-sm text-muted-foreground mt-1">{plugin.description}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {/* Event Types - Show skeleton during initial load */}
      {isLoading && eventTypes.length === 0 ? (
        <div className="space-y-3">
          <h3 className="font-semibold">Available Event Types</h3>
          <div className="grid md:grid-cols-3 gap-2">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <EventTypeSkeleton key={i} />
            ))}
          </div>
        </div>
      ) : eventTypes.length > 0 ? (
        <div className="space-y-3">
          <h3 className="font-semibold">Available Event Types</h3>
          <div className="grid md:grid-cols-3 gap-2">
            {eventTypes.map((event) => (
              <div key={event.id} className="p-2 bg-muted rounded-md">
                <div className="flex items-center gap-2">
                  <Zap className="h-3 w-3 text-primary" />
                  <p className="text-sm font-medium">{event.name}</p>
                </div>
                {event.description && (
                  <p className="text-xs text-muted-foreground mt-1">{event.description}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {/* Register New Hook */}
      <div className="space-y-4 p-4 border border-border rounded-lg">
        <h3 className="font-semibold">Register New Hook</h3>

        <div className="space-y-2">
          <label className="text-sm font-medium">Hook Name</label>
          <input
            type="text"
            value={hookName}
            onChange={(e) => setHookName(e.target.value)}
            className="w-full px-3 py-2 bg-background border border-border rounded-md"
            placeholder="My Hook"
            disabled={isLoading}
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Event Type</label>
          <select
            value={selectedEventType}
            onChange={(e) => setSelectedEventType(e.target.value)}
            className="w-full px-3 py-2 bg-background border border-border rounded-md"
            disabled={isLoading}
          >
            <option value="">Select event type...</option>
            {eventTypes.map((event) => (
              <option key={event.id} value={event.name}>
                {event.name}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Plugin</label>
          <select
            value={selectedPlugin}
            onChange={(e) => setSelectedPlugin(e.target.value)}
            className="w-full px-3 py-2 bg-background border border-border rounded-md"
            disabled={isLoading}
          >
            <option value="">Select plugin...</option>
            {plugins.filter(p => p.isEnabled).map((plugin) => (
              <option key={plugin.id} value={plugin.id}>
                {plugin.name} (v{plugin.version})
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Priority (lower = higher priority)</label>
          <input
            type="number"
            min="0"
            max="1000"
            value={priority}
            onChange={(e) => setPriority(parseInt(e.target.value))}
            className="w-full px-3 py-2 bg-background border border-border rounded-md"
            disabled={isLoading}
          />
        </div>

        <Button
          onClick={handleRegisterHook}
          disabled={isLoading || !hookName || !selectedEventType || !selectedPlugin}
          className="w-full"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Registering...
            </>
          ) : (
            <>
              <Plus className="h-4 w-4 mr-2" />
              Register Hook
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
