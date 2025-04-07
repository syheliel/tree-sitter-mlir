from tree_sitter import Language, Parser
from pathlib import Path
Language.build_library(
    'build/my-languages.so',
    ['./']
)

lang = Language('build/my-languages.so', 'mlir')
parser = Parser()
parser.set_language(lang)

seeds_path = list(Path('./corpus_success').glob('*.mlir'))
total_count = len(seeds_path)
success_count = 0
my_count = 0
for seed_path in seeds_path:
    with open(seed_path, 'r') as f:
        seed_code = f.read()

    try:
        tree = parser.parse(bytes(seed_code, 'utf-8'))
        if tree.root_node.has_error:
            if "arith" in seed_path.name:
                my_count += 1
                print(f"Error parsing {seed_path}")
                for child in tree.root_node.children:
                    print(child.has_error)
            
        else:
            success_count += 1
    except Exception as e:
        print(f"Error parsing {seed_path}: {e}")
print(f"Success rate: {success_count / total_count}: {success_count} / {total_count}") # 387 / 1332
print(f"My rate: {my_count / total_count}")