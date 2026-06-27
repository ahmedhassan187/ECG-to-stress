with open('src/main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, l in enumerate(lines[406:440], start=407):
    print(f"{i}: {repr(l)}")
