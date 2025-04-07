from tree_sitter import Node, Language, Parser
import sys
from loguru import logger
from pathlib import Path

def get_error_nodes(node: Node):
    def traverse_tree_for_errors(node: Node):
        for n in node.children:
            if n.type == "ERROR" or n.is_missing:
                yield n
            if n.has_error:
                # there is an error inside this node let's check inside
                yield from traverse_tree_for_errors(n)

    yield from traverse_tree_for_errors(node)


def print_error_line(er_line: int, padding: str, column_start, column_end, node_error):
    # print(f"{padding}{er_line}{padding}{file_lines[er_line-1]}")
    padding_with_line_number = " " * (len(f"{er_line}") + column_start-1)
    cursor_size = max(1, column_end - column_start)
    logger.error(
        f"{padding * 2}\033[31m{padding_with_line_number}{'~' * cursor_size}\033[0m")

    if node_error.has_error and node_error.is_missing:
        error_message = f"{node_error.sexp()[1:-1]}"
    else:
        unexpected_tokens = "".join(n.text.decode('utf-8')
                                    for n in node_error.children)
        error_message = f"Unexpected token(s): {unexpected_tokens}"
    logger.error(
        f"{padding * 2}\033[31m{padding_with_line_number}{error_message}:")


def print_error(root_node: Node, file_path: Path, error_type: str = "SYNTAX_ERROR"):
    padding = " " * 5
    for node_error in get_error_nodes(root_node):
        er_line = node_error.start_point[0]+1
        column_start = node_error.start_point[1] + 1
        column_end = node_error.end_point[1] + 1
        print(node_error.text)
        error_context = file_path.read_text().split('\n')[er_line-1]
        logger.error(
            f"\033[31m{error_type}\033[0m:  {node_error.sexp()[:]}")
        logger.error(
            f"{padding}in file: '{file_path}:{er_line}:{column_start}:{column_end}', line: {er_line}", end=", ")
        logger.error(
            f"from column {column_start} to {column_end}\n")
        logger.error(f"{padding}{error_context}")
        
        print_error_line(er_line, padding, column_start,
                             column_end, node_error)


def get_mlir_parser():
    parser = Parser()
    Language.build_library(
    'build/my-languages.so',
    ['./'])
    lang = Language('build/my-languages.so', 'mlir')
    parser.set_language(lang)
    return parser


