#!/usr/bin/env python3
from pathlib import Path

BASE = Path("/Users/macone/Desktop/Greta/cpas4-ui/src")

# Restore ActionNode.tsx
path = BASE / "components/agent-builder/nodes/ActionNode.tsx"
with open(path, 'r') as f:
    content = f.read()

# Fix the interface
content = content.replace(
    '''interface ActionNodeData {
//   label: string;
//   actionType?: string;
//   parameters?: Record<string, any>;
// }''',
    '''interface ActionNodeData {
  label: string;
  actionType?: string;
  parameters?: Record<string, any>;
}''')

with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed ActionNode.tsx")

# Restore AgentNode.tsx
path = BASE / "components/agent-builder/nodes/AgentNode.tsx"
with open(path, 'r') as f:
    content = f.read()

content = content.replace(
    '''interface AgentNodeData {
//   label: string;
//   agentType?: string;
//   model?: string;
//   temperature?: number;
//   maxTokens?: number;
//   systemPrompt?: string;
// }''',
    '''interface AgentNodeData {
  label: string;
  agentType?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  systemPrompt?: string;
}''')

with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed AgentNode.tsx")

# Restore ConditionNode.tsx
path = BASE / "components/agent-builder/nodes/ConditionNode.tsx"
with open(path, 'r') as f:
    content = f.read()

content = content.replace(
    '''interface ConditionNodeData {
//   label: string;
//   condition?: string;
//   operator?: string;
//   value?: string;
// }''',
    '''interface ConditionNodeData {
  label: string;
  condition?: string;
  operator?: string;
  value?: string;
}''')

with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed ConditionNode.tsx")

# Restore ToolNode.tsx
path = BASE / "components/agent-builder/nodes/ToolNode.tsx"
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # Fix the malformed interface
    if i >= 4 and i <= 9:
        if i == 4:
            new_lines.append('interface ToolNodeData {\n')
        elif i == 5:
            new_lines.append('  label: string;\n')
        elif i == 6:
            new_lines.append('  tools?: string[];\n')
        elif i == 7:
            new_lines.append('}\n')
        # Skip lines 8-9
    else:
        new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
print("✅ Fixed ToolNode.tsx")

# Restore MasterAgentContext.tsx
path = BASE / "components/agents/MasterAgentContext.tsx"
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
skip_lines = 0
for i, line in enumerate(lines):
    # Fix the _successMessage blocks - just comment them out entirely
    if 'const _successMessage:' in line:
        skip_lines = 5  # Skip the next 5 lines (the entire object)
        continue
    if skip_lines > 0:
        skip_lines -= 1
        continue
    new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
print("✅ Fixed MasterAgentContext.tsx")

print("\n✅ All files restored!")
