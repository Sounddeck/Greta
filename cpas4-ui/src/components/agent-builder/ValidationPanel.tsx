import React from 'react';

interface ValidationIssue {
  id?: string;
  message: string;
  source?: string;
  target?: string;
}

interface ValidationResultProps {
  result: {
    is_valid?: boolean;
    errors?: string[];
    node_errors?: ValidationIssue[];
    edge_errors?: ValidationIssue[];
    warnings?: string[];
  } | null;
}

export default function ValidationPanel({ result }: ValidationResultProps) {
  if (!result) return null;
  const { is_valid, errors = [], node_errors = [], edge_errors = [], warnings = [] } = result as any;

  const hasIssues = !is_valid || errors.length || node_errors.length || edge_errors.length || warnings.length;
  if (!hasIssues) return null;

  return (
    <div className="p-3 border-t border-white/10 bg-black/30 text-sm text-gray-200">
      <div className="font-semibold mb-2">Validation</div>
      {errors.length > 0 && (
        <div className="mb-2">
          <div className="text-red-400 font-medium">Errors</div>
          <ul className="list-disc list-inside text-red-300">
            {errors.map((e, i) => (
              <li key={`e-${i}`}>{e}</li>
            ))}
          </ul>
        </div>
      )}
      {node_errors.length > 0 && (
        <div className="mb-2">
          <div className="text-red-400 font-medium">Node Errors</div>
          <ul className="list-disc list-inside text-red-300">
            {node_errors.map((e, i) => (
              <li key={`ne-${i}`}>Node {e.id}: {e.message}</li>
            ))}
          </ul>
        </div>
      )}
      {edge_errors.length > 0 && (
        <div className="mb-2">
          <div className="text-red-400 font-medium">Edge Errors</div>
          <ul className="list-disc list-inside text-red-300">
            {edge_errors.map((e, i) => (
              <li key={`ee-${i}`}>{e.source} → {e.target}: {e.message}</li>
            ))}
          </ul>
        </div>
      )}
      {warnings.length > 0 && (
        <div className="mt-2">
          <div className="text-yellow-400 font-medium">Warnings</div>
          <ul className="list-disc list-inside text-yellow-300">
            {warnings.map((w, i) => (
              <li key={`w-${i}`}>{w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

