#!/usr/bin/env python3
"""Fix node component types - final correct approach"""
from pathlib import Path

BASE = Path("/Users/macone/Desktop/Greta/cpas4-ui/src")

# Fix all node files by removing NodeProps and using custom interface
node_files = {
    "ActionNode.tsx": "ActionNodeData",
    "AgentNode.tsx": "AgentNodeData",
    "ConditionNode.tsx": "ConditionNodeData",
    "ToolNode.tsx": "ToolNodeData"
}

for filename, data_type in node_files.items():
    path = BASE / f"components/agent-builder/nodes/{filename}"
    with open(path, 'r') as f:
        content = f.read()
    
    # Remove NodeProps import
    content = content.replace(
        "import { Handle, Position, type NodeProps } from '@xyflow/react';",
        "import { Handle, Position } from '@xyflow/react';"
    )
    
    # Change function signature to use custom props interface
    func_name = filename.replace('.tsx', '')
    old_sig = f"function {func_name}({{ data, selected }}: NodeProps<{{ data: {data_type} }}>) {{"
    new_sig = f"function {func_name}({{ data, selected }}: {{ data: {data_type}; selected?: boolean }}) {{"
    
    content = content.replace(old_sig, new_sig)
    
    # Remove type assertions since data is now properly typed
    content = content.replace(f'(data as {data_type}).', 'data.')
    content = content.replace(f'(data as {data_type})?', 'data?')
    
    with open(path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {filename}")

print("\n✅ All node types fixed with proper custom interfaces!")
