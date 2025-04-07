from tree_sitter import Language, Parser, Tree, Node
from pathlib import Path
from util import print_error
from typing import List
Language.build_library(
    'build/my-languages.so',
    ['./']
)

lang = Language('build/my-languages.so', 'mlir')
parser = Parser()
parser.set_language(lang)

seeds_path = list(Path('./corpus_success').glob('*.mlir'))
seeds_path = [seed for seed in seeds_path if 'affine' in seed.name]
total_count = len(seeds_path)
success_count = 0

def extract_error_part(tree:Tree,file_path:Path):
    """return the error source part of the tree"""
    print_error(tree.root_node,file_path)


success_seed_path = []
for seed_path in seeds_path:
    with open(seed_path, 'r') as f:
        seed_code = f.read()
        tree = parser.parse(bytes(seed_code, 'utf-8'))
        if tree.root_node.has_error:
            print(f"Error parsing {seed_path}")
            extract_error_part(tree,seed_path)
        else:
            success_count += 1
            success_seed_path.append(seed_path)

print(f"Success rate: {success_count / total_count}")
print(success_seed_path)