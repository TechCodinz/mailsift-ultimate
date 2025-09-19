#!/usr/bin/env python3
"""
Automated Linting Fixer for MailSift Ultimate
Fixes common linting issues automatically.
"""

import re
import os
from pathlib import Path

def fix_line_length(line: str, max_length: int = 79) -> str:
    """Fix line length by breaking long lines appropriately"""
    if len(line) <= max_length:
        return line
    
    # Don't break comments that are URLs or file paths
    if line.strip().startswith('#') and any(x in line for x in ['http', 'file://', 'ftp://']):
        return line
    
    # Don't break long strings that shouldn't be broken
    if line.strip().startswith(('"""', "'''")):
        return line
    
    # Break at logical points
    break_points = [
        ', ',  # After commas
        ' and ',  # Before "and"
        ' or ',   # Before "or"
        ' if ',   # Before "if"
        ' else ', # Before "else"
        ' = ',    # After assignments
        ' + ',    # After plus operators
        ' - ',    # After minus operators
        ' * ',    # After multiplication
        ' / ',    # After division
        ' % ',    # After modulo
        ' == ',   # After equality
        ' != ',   # After inequality
        ' <= ',   # After less than or equal
        ' >= ',   # After greater than or equal
        ' < ',    # After less than
        ' > ',    # After greater than
        ' in ',   # Before "in"
        ' not ',  # Before "not"
        ' with ', # Before "with"
        ' as ',   # Before "as"
        ' for ',  # Before "for"
        ' while ', # Before "while"
        ' return ', # Before "return"
        ' yield ',  # Before "yield"
        ' raise ',  # Before "raise"
        ' except ', # Before "except"
        ' finally ', # Before "finally"
    ]
    
    # Try to break at logical points
    for break_point in break_points:
        if break_point in line:
            parts = line.split(break_point)
            if len(parts) > 1:
                # Try to break at the first occurrence that keeps lines under limit
                for i in range(1, len(parts)):
                    left_part = break_point.join(parts[:i])
                    right_part = break_point.join(parts[i:])
                    
                    if (len(left_part) <= max_length and 
                        len(right_part) <= max_length):
                        # Calculate proper indentation
                        base_indent = len(line) - len(line.lstrip())
                        indent = ' ' * (base_indent + 4)
                        return f"{left_part}\n{indent}{right_part}"
    
    # If no good break point found, break at max_length
    if len(line) > max_length:
        base_indent = len(line) - len(line.lstrip())
        indent = ' ' * (base_indent + 4)
        return f"{line[:max_length]}\n{indent}{line[max_length:]}"
    
    return line

def fix_whitespace_issues(content: str) -> str:
    """Fix whitespace issues in file content"""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Remove trailing whitespace
        line = line.rstrip()
        
        # Fix line length
        line = fix_line_length(line)
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_file(file_path: str) -> bool:
    """Fix linting issues in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix whitespace issues
        fixed_content = fix_whitespace_issues(content)
        
        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"✅ Fixed: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Main function to fix all linting issues"""
    print("🔧 MailSift Ultimate - Automated Linting Fixer")
    print("=" * 50)
    
    # Files to fix
    files_to_fix = [
        "mailsift_ultimate.py",
        "crypto_payments.py",
        "ai_support.py",
        "desktop_app.py",
        "api_generator.py",
        "ultra_email_extractor.py",
        "ultra_web_scraper.py",
        "ultra_keyword_search.py",
        "ultra_error_handling.py",
        "ultra_monitoring.py",
        "ultra_performance.py",
        "build_installers.py",
        "create_simple_installer.py"
    ]
    
    fixed_count = 0
    total_count = 0
    
    for file_name in files_to_fix:
        if os.path.exists(file_name):
            total_count += 1
            if fix_file(file_name):
                fixed_count += 1
        else:
            print(f"⚠️  File not found: {file_name}")
    
    print(f"\n🎉 Linting Fix Complete!")
    print(f"📊 Fixed {fixed_count}/{total_count} files")
    
    if fixed_count == total_count:
        print("✅ All files fixed successfully!")
    else:
        print("⚠️  Some files had issues - check the output above")

if __name__ == "__main__":
    main()
