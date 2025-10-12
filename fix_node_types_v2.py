#!/usr/bin/env python3
"""Fix node component types properly"""
from pathlib import Path

BASE = Path("/Users/macone/Desktop/Greta/cpas4-ui/src")

# Fix all node files with proper typing
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
    
    # Change function signature to destructure data with proper typing
    old_sig = f"function {filename.replace('.tsx', '')}({{ data, selected }}: NodeProps<{data_type}>) {{"
    new_sig = f"function {filename.replace('.tsx', '')}({{ data, selected }}: NodeProps<{{ data: {data_type} }}>) {{"
    
    content = content.replace(old_sig, new_sig)
    
    # Add type assertion for data access
    # Replace "data." with "(data as {data_type})."
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        # Skip the function signature line
        if f"function {filename.replace('.tsx', '')}" in line:
            new_lines.append(line)
            continue
        
        # Add type assertion for data access
        if 'data.' in line or 'data?' in line:
            # Only add assertion if not already present
            if f'(data as {data_type})' not in line:
                line = line.replace('data.', f'(data as {data_type}).')
                line = line.replace('data?', f'(data as {data_type})?')
        
        new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    with open(path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {filename}")

# Fix MasterAgentContext - remove knowledgeAPI and fix errorMessage
path = BASE / "components/agents/MasterAgentContext.tsx"
with open(path, 'r') as f:
    content = f.read()

# Remove knowledgeAPI from import
content = content.replace(
    "import { taskAPI, promptAPI, knowledgeAPI } from '../../lib/api';",
    "import { taskAPI, promptAPI } from '../../lib/api';"
)

# Remove errorMessage reference
content = content.replace(
    "addMessage(currentAgentType, errorMessage);",
    "// Error handling removed - errorMessage was unused"
)

with open(path, 'w') as f:
    f.write(content)

print("✅ Fixed MasterAgentContext.tsx")
print("\n✅ All node types fixed!")
