with open('preLoginDashboard.js', 'r', encoding='utf-8') as f:
    text = f.read()

import re

# Look for split('').reverse().join('') or similar string reversals
print("Search for reverse in JS:")
for match in re.finditer(r'[a-zA-Z0-9_\-/]{5,}', text):
    word = match.group(0)
    rev = word[::-1]
    if 'rest' in rev or 'api' in rev or 'get' in rev or 'data' in rev or 'dashboard' in rev or 'PreLogin' in rev:
        print(f"Original: {word} -> Reversed: {rev}")

print("\n--- Printing lines containing ajax or fetch or rest ---")
for line in text.splitlines():
    if 'ajax' in line.lower() or 'fetch' in line.lower() or 'url' in line.lower() or 'rest' in line.lower() or 'reverse' in line.lower():
        print(line[:120])
