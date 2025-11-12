import React, { useEffect, useState } from 'react';
import { agentVersionAPI } from '../../lib/api';
import { Button } from '../ui/button';

interface Props {
  agentId: string;
}

export default function VersionSwitcher({ agentId }: Props) {
  const [versions, setVersions] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const res = await agentVersionAPI.list(agentId);
      setVersions(res.data || []);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (agentId) load();
  }, [agentId]);

  const clone = async () => {
    await agentVersionAPI.clone(agentId);
    await load();
  };

  const activate = async (versionId: string) => {
    await agentVersionAPI.activate(versionId);
    await load();
  };

  return (
    <div className="border-t border-white/10 bg-black/30 p-3 text-sm text-gray-200">
      <div className="flex items-center justify-between mb-2">
        <div className="font-semibold">Agent Versions</div>
        <Button onClick={clone} disabled={!agentId || loading}>Clone</Button>
      </div>
      <div className="space-y-1 text-xs">
        {versions.map((v) => (
          <div key={v.version_id} className="flex items-center justify-between bg-black/40 rounded px-2 py-1">
            <div>
              <div className="text-gray-100">{v.version_id}</div>
              <div className="text-gray-400">Status: {v.status || 'inactive'}</div>
            </div>
            <Button onClick={() => activate(v.version_id)} size="sm">Activate</Button>
          </div>
        ))}
        {versions.length === 0 && <div className="text-gray-400">No versions yet</div>}
      </div>
    </div>
  );
}

