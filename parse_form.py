import re

path = 'C:/Users/phj20/.gemini/antigravity/brain/0115d9a7-e846-4dc1-84b3-673739952152/.system_generated/steps/135/content.md'
txt = open(path, encoding='utf-8', errors='ignore').read()

print("Searching titles...")
titles = re.findall(r'<div class="[^"]*M7eMe[^"]*">(.*?)</div>', txt)
for t in titles:
    print("Field:", re.sub('<[^<]+?>', '', t))

headings = re.findall(r'role="heading"[^>]*>(.*?)</div>', txt)
for h in headings:
    print("Heading:", re.sub('<[^<]+?>', '', h))
