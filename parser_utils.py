from tree_sitter import Language, Parser

# 构建语言库
Language.build_library(
    'build/my-languages.so',
    [
        "./"
    ]
)

# 创建MLIR语言和解析器
MLIR_LANGUAGE = Language('./build/my-languages.so', 'mlir')
mlir_parser = Parser()
mlir_parser.set_language(MLIR_LANGUAGE)

# 导出解析器
__all__ = ["mlir_parser", "MLIR_LANGUAGE"] 