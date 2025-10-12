#!/usr/bin/env python3
import re
from pathlib import Path

BASE = Path("/Users/macone/Desktop/Greta/cpas4-ui/src")

# Fix ActionNode.tsx - restore import
path = BASE / "components/agent-builder/nodes/ActionNode.tsx"
with open(path, 'r') as f:
    content = f.read()
content = re.sub(r'^, \{ memo, useState \}', 'import { memo, useState }', content)
content = re.sub(r'type NodeProps<ActionNodeData>', 'NodeProps<ActionNodeData>', content)
with open(path, 'w') as f:
    f.write(content)
print(f"✅ Fixed ActionNode.tsx")

# Fix AgentNode.tsx
path = BASE / "components/agent-builder/nodes/AgentNode.tsx"
with open(path, 'r') as f:
    content = f.read()
content = re.sub(r'^, \{ memo, useState \}', 'import { memo, useState }', content)
content = re.sub(r'type NodeProps<AgentNodeData>', 'NodeProps<AgentNodeData>', content)
with open(path, 'w') as f:
    f.write(content)
print(f"✅ Fixed AgentNode.tsx")

# Fix ConditionNode.tsx
path = BASE / "components/agent-builder/nodes/ConditionNode.tsx"
with open(path, 'r') as f:
    content = f.read()
content = re.sub(r'^, \{ memo, useState \}', 'import { memo, useState }', content)
content = re.sub(r'type NodeProps<ConditionNodeData>', 'NodeProps<ConditionNodeData>', content)
with open(path, 'w') as f:
    f.write(content)
print(f"✅ Fixed ConditionNode.tsx")

# Fix ToolNode.tsx
path = BASE / "components/agent-builder/nodes/ToolNode.tsx"
with open(path, 'r') as f:
    content = f.read()
content = re.sub(r'^, \{ memo, useState \}', 'import { memo, useState }', content)
content = re.sub(r'type NodeProps<ToolNodeData>', 'NodeProps<ToolNodeData>', content)
with open(path, 'w') as f:
    f.write(content)
print(f"✅ Fixed ToolNode.tsx")

# Fix AgentBuilderWorkflow.tsx - fix nodeTypes
path = BASE / "components/agent-builder/AgentBuilderWorkflow.tsx"
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Fix nodeTypes object
    if 'agent: AgentNode,' in line:
        line = line.replace('agent: AgentNode,', 'agent: AgentNode as any,')
    if 'tool: ToolNode,' in line:
        line = line.replace('tool: ToolNode,', 'tool: ToolNode as any,')
    if 'condition: ConditionNode,' in line:
        line = line.replace('condition: ConditionNode,', 'condition: ConditionNode as any,')
    if 'action: ActionNode,' in line:
        line = line.replace('action: ActionNode,', 'action: ActionNode as any,')
    new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
print(f"✅ Fixed AgentBuilderWorkflow.tsx nodeTypes")

print("\n✅ All files fixed!")
