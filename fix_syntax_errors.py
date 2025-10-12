#!/usr/bin/env python3
from pathlib import Path

BASE = Path("/Users/macone/Desktop/Greta/cpas4-ui/src")

# Fix ToolNode.tsx - restore the interface
path = BASE / "components/agent-builder/nodes/ToolNode.tsx"
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Uncomment the interface lines
    if line.strip().startswith('// interface ToolNodeData'):
        line = line.replace('// ', '')
    if line.strip().startswith('// label:') or line.strip().startswith('// tools?:') or \
       line.strip().startswith('// category?:'):
        line = line.replace('// ', '')
    if line.strip() == '// }' and len(new_lines) > 0 and 'interface' in new_lines[-2]:
        line = line.replace('// ', '')
    new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
print("✅ Fixed ToolNode.tsx")

# Fix other node files too
for node_file in ["ActionNode.tsx", "AgentNode.tsx", "ConditionNode.tsx"]:
    path = BASE / f"components/agent-builder/nodes/{node_file}"
    with open(path, 'r') as f:
        content = f.read()
    
    # Uncomment all interface definitions
    content = content.replace('// interface', 'interface')
    content = content.replace('// label:', 'label:')
    content = content.replace('// actionType?:', 'actionType?:')
    content = content.replace('// parameters?:', 'parameters?:')
    content = content.replace('// agentType?:', 'agentType?:')
    content = content.replace('// model?:', 'model?:')
    content = content.replace('// temperature?:', 'temperature?:')
    content = content.replace('// maxTokens?:', 'maxTokens?:')
    content = content.replace('// systemPrompt?:', 'systemPrompt?:')
    content = content.replace('// condition?:', 'condition?:')
    content = content.replace('// operator?:', 'operator?:')
    content = content.replace('// value?:', 'value?:')
    
    with open(path, 'w') as f:
        f.write(content)
    print(f"✅ Fixed {node_file}")

# Fix MasterAgentContext - uncomment the message objects
path = BASE / "components/agents/MasterAgentContext.tsx"
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
in_commented_block = False
for i, line in enumerate(lines):
    # Uncomment the _successMessage blocks
    if '// const _successMessage:' in line:
        line = line.replace('// ', '')
        in_commented_block = True
    elif in_commented_block:
        if line.strip().startswith('//'):
            line = line.replace('// ', '', 1)
            if '};' in line:
                in_commented_block = False
        else:
            in_commented_block = False
    new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
print("✅ Fixed MasterAgentContext.tsx")

print("\n✅ All syntax errors fixed!")
