#!/usr/bin/env python3
from pathlib import Path

BASE = Path("/Users/macone/Desktop/Greta/cpas4-ui/src")

# Fix AgentBuilderWorkflow.tsx - fix nodeTypes properly
path = BASE / "components/agent-builder/AgentBuilderWorkflow.tsx"
with open(path, 'r') as f:
    content = f.read()

# Fix broken node names
content = content.replace('Agenttype Node', 'AgentNode as any')
content = content.replace('Tooltype Node', 'ToolNode as any')
content = content.replace('Conditiontype Node', 'ConditionNode as any')
content = content.replace('Actiontype Node', 'ActionNode as any')

with open(path, 'w') as f:
    f.write(content)

print("✅ Fixed AgentBuilderWorkflow.tsx")
