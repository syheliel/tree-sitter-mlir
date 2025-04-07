from parser_utils import mlir_parser

from tree_sitter import Node, Tree

import random
from . import Mutator

class ArithRandomAddOperator(Mutator):
    def mutate(self, tree: Tree):
        """
        随机在MLIR代码中添加新的算术算子，保持use关系
    """
        final_text = tree.text

        # 定义可添加的算子
        integer_operators = [
            'arith.addi', 'arith.subi', 'arith.muli', 'arith.divsi', 'arith.divui',
            'arith.ceildivsi', 'arith.ceildivui', 'arith.floordivsi',
            'arith.remsi', 'arith.remui', 'arith.andi', 'arith.ori', 'arith.xori',
            'arith.maxsi', 'arith.maxui', 'arith.minsi', 'arith.minui',
            'arith.shli', 'arith.shrsi', 'arith.shrui'
        ]

        float_operators = [
            'arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf', 'arith.remf',
            'arith.maximumf', 'arith.minimumf'
        ]

        # 查找所有可用的值（变量和常量）
        available_values = []
        value_types = {}  # 存储每个值的类型

        def _collect_available_values(node: Node):
            if node.type == "value_use":
                value_text = node.text.decode("utf-8")
                if value_text not in available_values:
                    available_values.append(value_text)

                    # 尝试找到该值的类型
                    parent = node.parent
                    if parent and parent.type == "arith_dialect":
                        # 查找类型注解
                        for child in parent.children:
                            if child.type == "_type_annotation":
                                type_text = child.text.decode("utf-8")
                                value_types[value_text] = type_text
                                break

            for child in node.children:
                _collect_available_values(child)

        _collect_available_values(tree.root_node)

        # 如果没有找到可用的值，则无法添加新算子
        if len(available_values) < 2:
            print("Not enough values available to add a new operator")
            return tree

        # 随机选择两个值作为新算子的输入
        lhs = random.choice(available_values)
        rhs = random.choice([v for v in available_values if v != lhs])

        # 确定值的类型
        lhs_type = value_types.get(lhs, "i32")  # 默认为i32
        rhs_type = value_types.get(rhs, "i32")  # 默认为i32

        # 选择算子类型
        if lhs_type.startswith("i") and rhs_type.startswith("i"):
            # 整数运算
            operator = random.choice(integer_operators)
            result_type = lhs_type  # 使用与输入相同的类型
        elif lhs_type.startswith("f") and rhs_type.startswith("f"):
            # 浮点运算
            operator = random.choice(float_operators)
            result_type = lhs_type  # 使用与输入相同的类型
        else:
            # 混合类型，默认使用整数运算
            operator = random.choice(integer_operators)
            result_type = "i32"

        # 生成新的算子代码
        new_var_name = f"%{len(available_values) + 1}"
        new_operator_code = f"    {new_var_name} = {operator} {lhs}, {rhs} : {result_type}\n"

        # 在函数体中插入新算子
        # 这里假设我们在函数体的最后一个return语句之前插入
        text_str = final_text.decode("utf-8")
        return_index = text_str.rindex("return")
        if return_index != -1:
            new_text = text_str[:return_index] + new_operator_code + text_str[return_index:]
            return mlir_parser.parse(new_text.encode("utf-8"))
        else:
            print("Could not find a suitable place to insert the new operator")
            return tree