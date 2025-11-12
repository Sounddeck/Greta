import { useEffect, useState } from 'react';
import { Activity, Cpu, HardDrive, Network } from 'lucide-react';

interface SystemResources {
  timestamp: string;
  cpu: {
    percent: number;
    count: number;
    frequency_mhz: number;
    status: string;
  };
  memory: {
    total_gb: number;
    used_gb: number;
    available_gb: number;
    percent: number;
    status: string;
  };
  swap: {
    total_gb: number;
    used_gb: number;
    percent: number;
    status: string;
  };
  disk: {
    total_gb: number;
    used_gb: number;
    free_gb: number;
    percent: number;
    status: string;
  };
  disk_io: {
    read_mb: number;
    write_mb: number;
    read_count: number;
    write_count: number;
  };
  network: {
    sent_mb: number;
    recv_mb: number;
    packets_sent: number;
    packets_recv: number;
  };
  system: {
    platform: string;
    platform_release: string;
    platform_version: string;
    architecture: string;
    processor: string;
    python_version: string;
    boot_time: string;
    uptime_hours: number;
    process_count: number;
  };
}

const getStatusColor = (status: string): string => {
  switch (status) {
    case 'good':
      return 'text-green-500';
    case 'moderate':
      return 'text-yellow-500';
    case 'warning':
      return 'text-orange-500';
    case 'critical':
      return 'text-red-500';
    default:
      return 'text-gray-500';
  }
};

const getStatusBgColor = (status: string): string => {
  switch (status) {
    case 'good':
      return 'bg-green-500/20';
    case 'moderate':
      return 'bg-yellow-500/20';
    case 'warning':
      return 'bg-orange-500/20';
    case 'critical':
      return 'bg-red-500/20';
    default:
      return 'bg-gray-500/20';
  }
};

export default function SystemStatusBar() {
  const [resources, setResources] = useState<SystemResources | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    const fetchResources = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/system/resources');
        if (!response.ok) {
          throw new Error('Failed to fetch system resources');
        }
        const data = await response.json();
        setResources(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    };

    // Initial fetch
    fetchResources();

    // Poll every 2 seconds
    const interval = setInterval(fetchResources, 2000);

    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <div className="bg-red-500/10 border-t border-red-500/20 px-4 py-2">
        <div className="flex items-center gap-2 text-red-500 text-sm">
          <Activity className="w-4 h-4" />
          <span>System monitoring unavailable: {error}</span>
        </div>
      </div>
    );
  }

  if (!resources) {
    return (
      <div className="bg-gray-800/50 border-t border-gray-700 px-4 py-2">
        <div className="flex items-center gap-2 text-gray-400 text-sm">
          <Activity className="w-4 h-4 animate-pulse" />
          <span>Loading system status...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900/50 border-t border-gray-700">
      {/* Compact View */}
      <div
        className="px-4 py-2 cursor-pointer hover:bg-gray-800/30 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between gap-2 text-xs">
          {/* CPU */}
          <div className="flex items-center gap-1.5">
            <Cpu className={`w-4 h-4 ${getStatusColor(resources.cpu.status)}`} />
            <span className="text-gray-400">CPU:</span>
            <span className={`font-mono ${getStatusColor(resources.cpu.status)}`}>
              {resources.cpu.percent}%
            </span>
          </div>

          {/* Memory */}
          <div className="flex items-center gap-1.5">
            <Activity className={`w-4 h-4 ${getStatusColor(resources.memory.status)}`} />
            <span className="text-gray-400">RAM:</span>
            <span className={`font-mono ${getStatusColor(resources.memory.status)}`}>
              {resources.memory.percent}%
            </span>
            <span className="text-gray-500 text-[10px]">
              ({resources.memory.used_gb.toFixed(1)}/{resources.memory.total_gb.toFixed(0)} GB)
            </span>
          </div>

          {/* Disk */}
          <div className="flex items-center gap-1.5">
            <HardDrive className={`w-4 h-4 ${getStatusColor(resources.disk.status)}`} />
            <span className="text-gray-400">Disk:</span>
            <span className={`font-mono ${getStatusColor(resources.disk.status)}`}>
              {resources.disk.percent}%
            </span>
          </div>

          {/* Network */}
          <div className="flex items-center gap-1.5">
            <Network className="w-4 h-4 text-blue-500" />
            <span className="text-gray-400">Net:</span>
            <span className="font-mono text-blue-500">
              ↑{(resources.network.sent_mb / 1024).toFixed(1)} GB
            </span>
            <span className="font-mono text-green-500">
              ↓{(resources.network.recv_mb / 1024).toFixed(1)} GB
            </span>
          </div>

          {/* Expand indicator */}
          <div className="text-gray-500 text-[10px]">
            {isExpanded ? '▼' : '▶'}
          </div>
        </div>
      </div>

      {/* Expanded View */}
      {isExpanded && (
        <div className="px-4 py-3 border-t border-gray-700 bg-gray-900/70">
          <div className="grid grid-cols-2 gap-2 text-xs">
            {/* CPU Details */}
            <div className={`p-3 rounded-lg ${getStatusBgColor(resources.cpu.status)}`}>
              <div className="flex items-center gap-2 mb-2">
                <Cpu className={`w-4 h-4 ${getStatusColor(resources.cpu.status)}`} />
                <span className="font-semibold text-gray-200">CPU</span>
              </div>
              <div className="space-y-1 text-gray-300">
                <div className="flex justify-between">
                  <span>Usage:</span>
                  <span className={`font-mono ${getStatusColor(resources.cpu.status)}`}>
                    {resources.cpu.percent}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Cores:</span>
                  <span className="font-mono">{resources.cpu.count}</span>
                </div>
                <div className="flex justify-between">
                  <span>Frequency:</span>
                  <span className="font-mono">{resources.cpu.frequency_mhz} MHz</span>
                </div>
              </div>
            </div>

            {/* Memory Details */}
            <div className={`p-3 rounded-lg ${getStatusBgColor(resources.memory.status)}`}>
              <div className="flex items-center gap-2 mb-2">
                <Activity className={`w-4 h-4 ${getStatusColor(resources.memory.status)}`} />
                <span className="font-semibold text-gray-200">Memory</span>
              </div>
              <div className="space-y-1 text-gray-300">
                <div className="flex justify-between">
                  <span>Usage:</span>
                  <span className={`font-mono ${getStatusColor(resources.memory.status)}`}>
                    {resources.memory.percent}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Used:</span>
                  <span className="font-mono">{resources.memory.used_gb.toFixed(2)} GB</span>
                </div>
                <div className="flex justify-between">
                  <span>Available:</span>
                  <span className="font-mono">{resources.memory.available_gb.toFixed(2)} GB</span>
                </div>
                <div className="flex justify-between">
                  <span>Total:</span>
                  <span className="font-mono">{resources.memory.total_gb.toFixed(2)} GB</span>
                </div>
              </div>
            </div>

            {/* Disk Details */}
            <div className={`p-3 rounded-lg ${getStatusBgColor(resources.disk.status)}`}>
              <div className="flex items-center gap-2 mb-2">
                <HardDrive className={`w-4 h-4 ${getStatusColor(resources.disk.status)}`} />
                <span className="font-semibold text-gray-200">Disk</span>
              </div>
              <div className="space-y-1 text-gray-300">
                <div className="flex justify-between">
                  <span>Usage:</span>
                  <span className={`font-mono ${getStatusColor(resources.disk.status)}`}>
                    {resources.disk.percent}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Used:</span>
                  <span className="font-mono">{resources.disk.used_gb.toFixed(2)} GB</span>
                </div>
                <div className="flex justify-between">
                  <span>Free:</span>
                  <span className="font-mono">{resources.disk.free_gb.toFixed(2)} GB</span>
                </div>
                <div className="flex justify-between">
                  <span>Total:</span>
                  <span className="font-mono">{resources.disk.total_gb.toFixed(2)} GB</span>
                </div>
              </div>
            </div>

            {/* System Info */}
            <div className="p-3 rounded-lg bg-gray-800/50">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-4 h-4 text-gray-400" />
                <span className="font-semibold text-gray-200">System</span>
              </div>
              <div className="space-y-1 text-gray-300">
                <div className="flex justify-between">
                  <span>Platform:</span>
                  <span className="font-mono">{resources.system.platform}</span>
                </div>
                <div className="flex justify-between">
                  <span>Architecture:</span>
                  <span className="font-mono">{resources.system.architecture}</span>
                </div>
                <div className="flex justify-between">
                  <span>Uptime:</span>
                  <span className="font-mono">{resources.system.uptime_hours.toFixed(1)}h</span>
                </div>
                <div className="flex justify-between">
                  <span>Processes:</span>
                  <span className="font-mono">{resources.system.process_count}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

