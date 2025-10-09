#!/bin/bash

# Script to replace ALL grey button backgrounds with blue
# This will change bg-gray-500, bg-gray-600, bg-gray-700, bg-gray-800 to blue equivalents

echo "🔧 Fixing all grey buttons in GRETA frontend..."

# Navigate to frontend directory
cd greta-frontend/src

# Find all .tsx files and replace grey backgrounds with blue
# bg-gray-500 -> bg-blue-600
# bg-gray-600 -> bg-blue-700
# bg-gray-700 -> bg-blue-800
# bg-gray-800 -> bg-slate-800 (for dark backgrounds)

# Also update hover states
# hover:bg-gray-600 -> hover:bg-blue-700
# hover:bg-gray-700 -> hover:bg-blue-800
# hover:bg-gray-800 -> hover:bg-blue-900

echo "Replacing bg-gray-500 with bg-blue-600..."
find . -name "*.tsx" -type f -exec sed -i '' 's/bg-gray-500/bg-blue-600/g' {} +

echo "Replacing bg-gray-600 with bg-blue-700..."
find . -name "*.tsx" -type f -exec sed -i '' 's/bg-gray-600/bg-blue-700/g' {} +

echo "Replacing bg-gray-700 with bg-blue-800..."
find . -name "*.tsx" -type f -exec sed -i '' 's/bg-gray-700/bg-blue-800/g' {} +

echo "Replacing bg-gray-800 with bg-slate-800..."
find . -name "*.tsx" -type f -exec sed -i '' 's/bg-gray-800/bg-slate-800/g' {} +

echo "Replacing hover:bg-gray-500 with hover:bg-blue-700..."
find . -name "*.tsx" -type f -exec sed -i '' 's/hover:bg-gray-500/hover:bg-blue-700/g' {} +

echo "Replacing hover:bg-gray-600 with hover:bg-blue-800..."
find . -name "*.tsx" -type f -exec sed -i '' 's/hover:bg-gray-600/hover:bg-blue-800/g' {} +

echo "Replacing hover:bg-gray-700 with hover:bg-blue-900..."
find . -name "*.tsx" -type f -exec sed -i '' 's/hover:bg-gray-700/hover:bg-blue-900/g' {} +

echo "Replacing hover:bg-gray-800 with hover:bg-slate-900..."
find . -name "*.tsx" -type f -exec sed -i '' 's/hover:bg-gray-800/hover:bg-slate-900/g' {} +

echo ""
echo "✅ All grey buttons have been changed to blue!"
echo ""
echo "Summary of changes:"
echo "  bg-gray-500 → bg-blue-600"
echo "  bg-gray-600 → bg-blue-700"
echo "  bg-gray-700 → bg-blue-800"
echo "  bg-gray-800 → bg-slate-800"
echo ""
echo "  hover:bg-gray-500 → hover:bg-blue-700"
echo "  hover:bg-gray-600 → hover:bg-blue-800"
echo "  hover:bg-gray-700 → hover:bg-blue-900"
echo "  hover:bg-gray-800 → hover:bg-slate-900"
echo ""
echo "🎨 Your buttons now match the blue theme!"

