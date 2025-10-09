#!/bin/bash

# Fix all remaining grey elements in GRETA frontend

cd greta-frontend/src

echo "🔧 Fixing remaining grey elements..."

# Fix all bg-gray-400 to bg-blue-400
find . -name "*.tsx" -type f -exec sed -i '' 's/bg-gray-400/bg-blue-400/g' {} +

# Fix all text-gray-400 in status badges (but keep text-gray-400 for labels)
# We'll do this more carefully - only in getStatusColor, getTypeColor, etc functions
find . -name "*.tsx" -type f -exec sed -i '' "s/'text-gray-400 bg-blue-400/'text-blue-400 bg-blue-400/g" {} +

# Fix bg-white/5 and bg-white/10
find . -name "*.tsx" -type f -exec sed -i '' 's/bg-white\/5/bg-blue-500\/10/g' {} +
find . -name "*.tsx" -type f -exec sed -i '' 's/bg-white\/10/bg-blue-500\/15/g' {} +
find . -name "*.tsx" -type f -exec sed -i '' 's/hover:bg-white\/5/hover:bg-blue-500\/10/g' {} +
find . -name "*.tsx" -type f -exec sed -i '' 's/hover:bg-white\/10/hover:bg-blue-500\/15/g' {} +

echo "✅ All grey elements fixed!"
echo ""
echo "Changes made:"
echo "  bg-gray-400 → bg-blue-400"
echo "  bg-white/5 → bg-blue-500/10"
echo "  bg-white/10 → bg-blue-500/15"
echo "  hover:bg-white/5 → hover:bg-blue-500/10"
echo "  hover:bg-white/10 → hover:bg-blue-500/15"
echo "  text-gray-400 (in status badges) → text-blue-400"

