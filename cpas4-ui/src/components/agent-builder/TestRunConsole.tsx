import React, { useRef, useState } from 'react';
import { agentRunAPI } from '../../lib/api';
import { Button } from '../ui/button';
import { Input } from '../ui/input';

interface Props {
  agentIdHint?: string; // optional known agent id to start with
}

export default function TestRunConsole({ agentIdHint }: Props) {
  const [agentId, setAgentId] = useState(agentIdHint || '');
  const [prompt, setPrompt] = useState('');
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<any[]>([]);
  const [output, setOutput] = useState<any | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const handleStart = async () => {
    if (!agentId) return;
    if (!prompt) return;
    setRunning(true);
    setLogs([]);
    setOutput(null);
    try {
      const res = await agentRunAPI.start(agentId, { prompt });
      const data = res.data || {};
      const runId = data.run_id;
      // Seed with any existing logs
      setLogs(data.logs || []);
      setOutput(data.output || null);

      // Open WebSocket for streaming
      const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';
      const wsUrl = API_BASE_URL.replace('http', 'ws') + `/agent-builder/runs/${runId}/ws`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      ws.onmessage = (evt) => {
        try {
          const ev = JSON.parse(evt.data);
          if (ev.error) {
            setLogs((l) => [...l, { level: 'error', message: ev.error }]);
            return;
          }
          if (ev.logs) {
            // snapshot
            setLogs(ev.logs);
          }
          if (ev.message) {
            setLogs((l) => [...l, { level: ev.level || 'info', message: ev.message }]);
          }
          if (ev.output) {
            setOutput(ev.output);
          }
          if (ev.status && ['completed', 'failed', 'canceled'].includes(ev.status)) {
            ws.close();
            setRunning(false);
          }
        } catch {}
      };
      ws.onerror = () => {
        setLogs((l) => [...l, { level: 'error', message: 'WebSocket error' }]);
      };
      ws.onclose = () => {
        setRunning(false);
      };
    } catch (e) {
      setLogs((l) => [...l, { level: 'error', message: 'Run failed' }]);
      setRunning(false);
    }
  };

  return (
    <div className="border-t border-white/10 bg-black/30 p-3 text-sm text-gray-200">
      <div className="font-semibold mb-2">Test Run</div>
      <div className="flex gap-2 mb-2">
        <Input
          value={agentId}
          onChange={(e) => setAgentId(e.target.value)}
          placeholder="Agent ID (from Deploy)"
        />
        <Input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a prompt to test"
        />
        <Button onClick={handleStart} disabled={!agentId || !prompt || running}>
          {running ? 'Running...' : 'Start'}
        </Button>
      </div>
      {logs.length > 0 && (
        <div className="mb-2 max-h-40 overflow-auto bg-black/40 rounded p-2 text-xs">
          {logs.map((l, i) => (
            <div key={i} className={l.level === 'error' ? 'text-red-300' : 'text-gray-300'}>
              [{l.level || 'info'}] {l.message}
            </div>
          ))}
        </div>
      )}
      {output && (
        <div className="bg-black/40 rounded p-2 text-xs text-green-300">
          <div className="font-medium mb-1">Output</div>
          <pre className="whitespace-pre-wrap">{JSON.stringify(output, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

