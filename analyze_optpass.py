import json

# Read the dialect list
with open('knowledge_base/dialect_list.json', 'r') as f:
    dialect_data = json.load(f)
    dialects = dialect_data['dialects']

# Read the optpass list
with open('knowledge_base/optpass_list.json', 'r') as f:
    optpasses = json.load(f)

# Dictionary to store counts
dialect_counts = {dialect: 0 for dialect in dialects}

# Count passes for each dialect
for pass_info in optpasses:
    pass_name = pass_info['name'].lower()
    for dialect in dialects:
        if dialect.lower() in pass_name:
            dialect_counts[dialect] += 1

# Print results
print("\nAnalysis of OptPasses per Dialect:")
print("-" * 40)
print(f"{'Dialect':<20} {'Number of Passes':<20}")
print("-" * 40)

# Sort by number of passes (descending)
sorted_dialects = sorted(dialect_counts.items(), key=lambda x: x[1], reverse=True)

for dialect, count in sorted_dialects:
    if count > 0:  # Only show dialects that have passes
        print(f"{dialect:<20} {count:<20}")

# Print total
total_passes = sum(dialect_counts.values())
print("-" * 40)
print(f"Total unique passes analyzed: {len(optpasses)}")
print(f"Total dialect-related passes: {total_passes}")

# Print affine-related passes
print("\nAffine-related passes:")
print("-" * 40)
for pass_info in optpasses:
    pass_name = pass_info['name'].lower()
    if 'affine' in pass_name:
        print(f"- {pass_info['name']}")

