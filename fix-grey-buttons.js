#!/usr/bin/env node

/**
 * Script to replace all grey buttons with darker blue buttons
 * to match the GRETA theme
 */

const fs = require('fs');
const path = require('path');

const replacements = [
  // Grey button backgrounds
  { from: /bg-gray-600(\s+hover:bg-gray-700)?/g, to: 'bg-blue-700 hover:bg-blue-800' },
  { from: /bg-gray-700(\s+hover:bg-gray-800)?/g, to: 'bg-blue-800 hover:bg-blue-900' },
  { from: /bg-gray-800(\s+hover:bg-gray-900)?/g, to: 'bg-slate-800 hover:bg-slate-900' },
  
  // Grey backgrounds for modals and containers (keep these as slate for contrast)
  // These are intentionally not changed as they provide good contrast
];

function processFile(filePath) {
  try {
    let content = fs.readFileSync(filePath, 'utf8');
    let modified = false;
    
    replacements.forEach(({ from, to }) => {
      const newContent = content.replace(from, to);
      if (newContent !== content) {
        content = newContent;
        modified = true;
      }
    });
    
    if (modified) {
      fs.writeFileSync(filePath, content, 'utf8');
      console.log(`✅ Fixed: ${filePath}`);
      return true;
    }
    
    return false;
  } catch (error) {
    console.error(`❌ Error processing ${filePath}:`, error.message);
    return false;
  }
}

function processDirectory(dirPath) {
  const files = fs.readdirSync(dirPath);
  let totalFixed = 0;
  
  files.forEach(file => {
    const filePath = path.join(dirPath, file);
    const stat = fs.statSync(filePath);
    
    if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
      totalFixed += processDirectory(filePath);
    } else if (file.endsWith('.tsx') || file.endsWith('.jsx')) {
      if (processFile(filePath)) {
        totalFixed++;
      }
    }
  });
  
  return totalFixed;
}

// Main execution
const frontendPath = path.join(__dirname, 'greta-frontend', 'src');

if (!fs.existsSync(frontendPath)) {
  console.error('❌ Frontend directory not found:', frontendPath);
  process.exit(1);
}

console.log('🔧 Fixing grey buttons in GRETA frontend...\n');
const fixed = processDirectory(frontendPath);
console.log(`\n✨ Complete! Fixed ${fixed} files.`);

