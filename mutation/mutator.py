from abc import ABC, abstractmethod
from tree_sitter import Tree

class Mutator(ABC):
    @abstractmethod
    def mutate(self, tree: Tree) -> Tree:
        pass
