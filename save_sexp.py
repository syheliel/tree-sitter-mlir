from tree_sitter import Language, Parser, Node
import sys
Language.build_library(
    'build/my-languages.so',
    ['./']
)

lang = Language('build/my-languages.so', 'mlir')
parser = Parser()
parser.set_language(lang)

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(input_file, 'r') as f:
        seed_code = f.read()
    tree = parser.parse(bytes(seed_code, 'utf-8'))
    with open(output_file, 'w') as f:
        # Format the s-expression with indentation
        def format_sexp(node:Node, level=0):
            if node.type == "comment":
                return ""
            indent = "  " * level
            result = f"{indent}({node.type} start_line:{node.start_point[0]+1} end_line:{node.end_point[0]+1}"
            if node.text:
                text = node.text.decode('utf-8').replace('\n', '\\n')
                result += f" text:'{text}'"
            if len(node.children) > 0:
                result += "\n"
                for child in node.children:
                    child_result = format_sexp(child, level + 1)
                    if child_result:  # Only add non-empty results
                        result += child_result
                result += indent
            result += ")\n"
            return result
        formatted_sexp = format_sexp(tree.root_node)
        f.write(formatted_sexp)