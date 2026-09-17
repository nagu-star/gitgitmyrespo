import re

with open('poptable_decoded.js', 'r', encoding='utf-8') as f:
    text = f.read()

# Search for /rest/ or API endpoints
rest_urls = re.findall(r'[\"\'\`](\/rest\/[^\s\"\'\`]+)[\"\'\`]', text)
print("REST URLs found in poptable.js:")
for u in set(rest_urls):
    print(" - ", u)

# Search for DataTable columns or field names
cols = re.findall(r'[\"\'\`]data[\"\'`]\s*:\s*[\"\'\`]([a-zA-Z0-9_]+)[\"\'\`]', text)
print("\nDataTable Column fields found:")
for c in sorted(list(set(cols))):
    print(" - ", c)

# Search for reversed strings containing rest/
words = re.findall(r'[a-zA-Z0-9_/\.]{4,}', text)
rev_rest = set()
for w in words:
    rev = w[::-1]
    if 'rest/' in rev:
        rev_rest.add(rev)
    if 'rest/' in w:
        rev_rest.add(w)

print("\nReversed REST URLs found:")
for r in rev_rest:
    print(" - ", r)
