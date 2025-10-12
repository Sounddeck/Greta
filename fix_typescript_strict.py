#!/usr/bin/env python3
"""Fix TypeScript strict mode errors properly"""
from pathlib import Path

BASE = Path("/Users/macone/Desktop/Greta/cpas4-ui/src")

# Fix ActionNode.tsx
path = BASE / "components/agent-builder/nodes/ActionNode.tsx"
with open(path, 'r') as f:
    content = f.read()

# Add NodeProps import and use proper typing
content = content.replace(
    "import { Handle, Position } from '@xyflow/react';",
    "import { Handle, Position, type NodeProps } from '@xyflow/react';"
)
content = content.replace(
    "function ActionNode({ data, selected }: any /* NodeProps<ActionNodeData> */ ) {",
    "function ActionNode({ data, selected }: NodeProps<ActionNodeData>) {"
)

with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed ActionNode.tsx")

# Fix AgentNode.tsx
path = BASE / "components/agent-builder/nodes/AgentNode.tsx"
with open(path, 'r') as f:
    content = f.read()

content = content.replace(
    "import { Handle, Position } from '@xyflow/react';",
    "import { Handle, Position, type NodeProps } from '@xyflow/react';"
)
content = content.replace(
    "function AgentNode({ data, selected }: any /* NodeProps<AgentNodeData> */ ) {",
    "function AgentNode({ data, selected }: NodeProps<AgentNodeData>) {"
)

with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed AgentNode.tsx")

# Fix ConditionNode.tsx
path = BASE / "components/agent-builder/nodes/ConditionNode.tsx"
with open(path, 'r') as f:
    content = f.read()

content = content.replace(
    "import { Handle, Position } from '@xyflow/react';",
    "import { Handle, Position, type NodeProps } from '@xyflow/react';"
)
content = content.replace(
    "function ConditionNode({ data, selected }: any /* NodeProps<ConditionNodeData> */ ) {",
    "function ConditionNode({ data, selected }: NodeProps<ConditionNodeData>) {"
)

with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed ConditionNode.tsx")

# Fix ToolNode.tsx
path = BASE / "components/agent-builder/nodes/ToolNode.tsx"
with open(path, 'r') as f:
    content = f.read()

content = content.replace(
    "import { Handle, Position } from '@xyflow/react';",
    "import { Handle, Position, type NodeProps } from '@xyflow/react';"
)
content = content.replace(
    "function ToolNode({ data, selected }: any /* NodeProps<ToolNodeData> */ ) {",
    "function ToolNode({ data, selected }: NodeProps<ToolNodeData>) {"
)

with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed ToolNode.tsx")

# Fix MasterAgentContext.tsx - remove unused knowledgeAPI and errorMessage
path = BASE / "components/agents/MasterAgentContext.tsx"
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
skip_error_message = False
for i, line in enumerate(lines):
    # Remove knowledgeAPI from import
    if 'knowledgeAPI,' in line:
        line = line.replace('knowledgeAPI,', '')
    
    # Skip errorMessage declarations and their objects
    if 'const errorMessage:' in line:
        skip_error_message = True
        continue
    
    if skip_error_message:
        if line.strip() == '};':
            skip_error_message = False
        continue
    
    new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
print("✅ Fixed MasterAgentContext.tsx")

print("\n✅ All TypeScript strict mode errors fixed!")
