#!/usr/bin/env python3
"""
Comprehensive script to fix ALL grey elements in GRETA frontend
"""

import os
import re
from pathlib import Path

# Define ALL replacements
REPLACEMENTS = {
    # Solid grey backgrounds
    'bg-gray-400/20': 'bg-blue-400/20',
    'bg-gray-400': 'bg-blue-400',
    'bg-gray-500/20': 'bg-blue-500/20',
    'bg-gray-500': 'bg-blue-500',
    'bg-gray-600': 'bg-blue-600',
    'bg-gray-700': 'bg-blue-700',
    'bg-gray-800': 'bg-blue-800',
    'bg-gray-900': 'bg-blue-900',
    
    # White backgrounds that appear grey
    'bg-white/5': 'bg-blue-500/10',
    'bg-white/10': 'bg-blue-500/15',
    'hover:bg-white/5': 'hover:bg-blue-500/10',
    'hover:bg-white/10': 'hover:bg-blue-500/15',
    
    # Hover states
    'hover:bg-gray-400': 'hover:bg-blue-400',
    'hover:bg-gray-500': 'hover:bg-blue-500',
    'hover:bg-gray-600': 'hover:bg-blue-600',
    'hover:bg-gray-700': 'hover:bg-blue-700',
    'hover:bg-gray-800': 'hover:bg-blue-800',
    'hover:bg-gray-900': 'hover:bg-blue-900',
}

def process_file(file_path):
    """Process a single file and replace grey backgrounds with blue"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = []
        
        # Apply all replacements
        for old, new in REPLACEMENTS.items():
            if old in content:
                count = content.count(old)
                content = content.replace(old, new)
                changes_made.append(f"  {old} → {new} ({count} times)")
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return changes_made
        
        return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    """Main function to process all files"""
    frontend_dir = Path('greta-frontend/src')
    
    if not frontend_dir.exists():
        print(f"Error: {frontend_dir} does not exist!")
        return
    
    print("🔧 Fixing ALL grey elements in GRETA frontend...")
    print("=" * 80)
    
    total_files_changed = 0
    total_changes = 0
    
    # Process all .tsx files
    for file_path in frontend_dir.rglob('*.tsx'):
        changes = process_file(file_path)
        if changes:
            total_files_changed += 1
            total_changes += len(changes)
            print(f"\n✅ {file_path.relative_to(frontend_dir)}:")
            for change in changes:
                print(change)
    
    # Process all .ts files (excluding .tsx)
    for file_path in frontend_dir.rglob('*.ts'):
        if file_path.name.endswith('.tsx'):
            continue
        changes = process_file(file_path)
        if changes:
            total_files_changed += 1
            total_changes += len(changes)
            print(f"\n✅ {file_path.relative_to(frontend_dir)}:")
            for change in changes:
                print(change)
    
    print("\n" + "=" * 80)
    print(f"✨ Complete!")
    print(f"📊 Files modified: {total_files_changed}")
    print(f"🔄 Total replacements: {total_changes}")
    print("\n🎨 All grey elements have been changed to blue!")

if __name__ == '__main__':
    main()

