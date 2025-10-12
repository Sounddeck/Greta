#!/usr/bin/env python3
from pathlib import Path

BASE = Path("/Users/macone/Desktop/Greta/cpas4-ui/src")
path = BASE / "components/agent-builder/nodes/ToolNode.tsx"

with open(path, 'r') as f:
    content = f.read()

# Fix the missing const declaration
content = content.replace(
    '''interface ToolNodeData {
  label: string;
  tools?: string[];
}
  { name: 'Code Execution', category: 'Development' },''',
    '''interface ToolNodeData {
  label: string;
  tools?: string[];
}

const availableTools = [
  { name: 'Code Execution', category: 'Development' },''')

with open(path, 'w') as f:
    f.write(content)

print("✅ Fixed ToolNode.tsx")
