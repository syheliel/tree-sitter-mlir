from parser_utils import mlir_parser

from tree_sitter import Node, Tree

import random


def arith_random_replace_operator(tree: Tree):
    """
    随机替换MLIR中的算术算子，保持输入输出类型一致
    """
    final_text = tree.text

    # 定义可替换的算子组
    # 整数运算算子组
    integer_operators = [
        'arith.addi', 'arith.subi', 'arith.muli', 'arith.divsi', 'arith.divui',
        'arith.ceildivsi', 'arith.ceildivui', 'arith.floordivsi',
        'arith.remsi', 'arith.remui', 'arith.andi', 'arith.ori', 'arith.xori',
        'arith.maxsi', 'arith.maxui', 'arith.minsi', 'arith.minui',
        'arith.shli', 'arith.shrsi', 'arith.shrui'
    ]

    # 浮点运算算子组
    float_operators = [
        'arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf', 'arith.remf',
        'arith.maximumf', 'arith.minimumf'
    ]

    # 比较算子组
    compare_operators = ['arith.cmpi', 'arith.cmpf']

    # 类型转换算子组
    cast_operators = [
        'arith.extf', 'arith.extsi', 'arith.extui', 'arith.fptosi', 'arith.fptoui',
        'arith.index_cast', 'arith.index_castui', 'arith.sitofp', 'arith.uitofp',
        'arith.bitcast', 'arith.truncf', 'arith.trunci'
    ]

    def _random_replace_operator(node: Node):
        if node.type == "arith_dialect":
            original_text = node.text.decode("utf-8")

            # 确定当前算子的类型
            current_operator = None
            for op in integer_operators + float_operators + compare_operators + cast_operators:
                if op in original_text:
                    current_operator = op
                    break

            if current_operator:
                # 根据当前算子类型选择可替换的算子组
                if current_operator in integer_operators:
                    new_operator = random.choice([op for op in integer_operators if op != current_operator])
                elif current_operator in float_operators:
                    new_operator = random.choice([op for op in float_operators if op != current_operator])
                elif current_operator in compare_operators:
                    new_operator = random.choice([op for op in compare_operators if op != current_operator])
                elif current_operator in cast_operators:
                    new_operator = random.choice([op for op in cast_operators if op != current_operator])
                else:
                    return  # 不替换未知类型的算子

                # 替换算子
                new_text = original_text.replace(current_operator, new_operator)

                # 更新节点
                new_end_byte = node.start_byte + len(new_text)
                new_end_point = (node.end_point[0], node.end_point[1] + len(new_text))
                node.edit(node.start_byte, node.end_byte, new_end_byte, node.start_point, node.end_point, new_end_point)

                # 更新文本
                nonlocal final_text
                final_text = final_text[:node.start_byte] + new_text.encode("utf-8") + final_text[node.end_byte:]
                print(f"Replaced {current_operator} with {new_operator}")

        for child in node.children:
            _random_replace_operator(child)

    _random_replace_operator(tree.root_node)
    return mlir_parser.parse(final_text)