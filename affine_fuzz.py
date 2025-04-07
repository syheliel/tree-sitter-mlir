from pathlib import Path
from mutation import AffineDataCopyGenerate
from util import get_mlir_parser
from tree_sitter import Tree
from loguru import logger
if __name__ == '__main__':
    parser = get_mlir_parser()
    seeds_path = [Path('corpus_success/lower-affine-to-vector.mlir'), Path('corpus_success/affine-walk.mlir'), Path('corpus_success/affine.mlir'), Path('corpus_success/vectorize_affine_apply.mlir'), Path('corpus_success/affine-expand-index-ops-as-affine.mlir'), Path('corpus_success/lower-affine.mlir')]

    trees:list[Tree] = []
    for seed_path in seeds_path:
        with open(seed_path, 'r') as f:
            seed_code = f.read()
            tree = parser.parse(bytes(seed_code, 'utf-8'))
            if tree.root_node.has_error:
                logger.error(f"Error parsing {seed_path}")
            else:
                trees.append(tree)
    
    opt_guided_mutator = [AffineDataCopyGenerate()]
    random_mutator = [ArithRandomAddOperator(), ArithRandomReplaceOperator()]
    for seed in trees:
        seed_code = AffineDataCopyGenerate().mutate(seed)
        if seed_code.root_node.has_error:
            logger.error(f"Error parsing {seed_path}")
        else:
            print(seed_code)
                
            