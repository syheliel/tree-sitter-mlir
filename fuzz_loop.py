from tree_sitter import Node, Tree
from parser_utils import mlir_parser
from mutation import arith_random_add_operator, arith_random_replace_operator

start = mlir_parser.parse(bytes("""
// Example MLIR file for testing the analyzer
module {
  func.func @add(%a: i32, %b: i32) -> i32 {
    %0 = arith.addi %a, %b : i32
    return %0 : i32
  }
}""", "utf-8"))

end = start
for i in range(50): 
    end = arith_random_add_operator(end)
    print(end.root_node.is_error)
    print(end.text)