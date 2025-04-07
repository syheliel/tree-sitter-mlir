from parser_utils import mlir_parser

from tree_sitter import Node, Tree
from mutation import Mutator
import random


class AffineDataCopyGenerate1(Mutator):
    def mutate(self, tree: Tree):
        """
        Promote memory operations to faster memory spaces by allocating buffers
        in higher memory spaces and using them for temporary computations.
        Focuses on promoting whole arrays rather than individual access points.
        """
        final_text = tree.text

        # Configuration parameters
        fast_memory_space = random.randint(1, 4)
        
        # Find memref.alloc operations as candidates for promotion
        memref_allocs = []
        
        def _find_memref_allocs(node: Node):
            if node.type in ["mlir_operation", "mlir_operation_body"]:
                text = node.text.decode("utf-8")
                if "memref.alloc" in text:
                    memref_allocs.append(node)
            
            for child in node.children:
                _find_memref_allocs(child)
        
        _find_memref_allocs(tree.root_node)
        
        # If no memref allocations found, return the original tree
        if not memref_allocs:
            print("No memref allocations found, cannot apply memory promotion")
            return tree
        
        # Select a random memref allocation to transform
        target_alloc = random.choice(memref_allocs)
        target_text = target_alloc.text.decode("utf-8")
        
        # Parse the allocation to extract memref name and type
        try:
            lines = target_text.strip().split('\n')
            alloc_line = next((line for line in lines if "memref.alloc" in line), "")
            
            if not alloc_line:
                return tree
            
            # Extract allocated memref and its type
            # Format: %memref = memref.alloc(...) : memref_type
            parts = alloc_line.split('=')
            if len(parts) < 2:
                return tree
                
            memref_var = parts[0].strip()
            
            # Extract the type
            type_part = parts[1].split(':')
            if len(type_part) < 2:
                return tree
                
            memref_type = type_part[1].strip()
            
            # Check if already in a non-default memory space
            if "memspace" in memref_type:
                # Already in a specific memory space, no need to promote
                return tree
        except:
            # If parsing fails, return the original tree
            print("Failed to parse the memref allocation")
            return tree
        
        # Generate the promoted memref type with fast memory space
        if ">" in memref_type:
            promoted_type = memref_type.replace(">", f", memref.memory_space<{fast_memory_space}>>")
        else:
            # Handle the case where there is no closing bracket
            promoted_type = f"{memref_type}<memref.memory_space<{fast_memory_space}>>"
        
        # Generate the corresponding allocation in the fast memory space
        promoted_var = f"%promoted_{memref_var[1:]}"
        promoted_alloc = alloc_line.replace(memref_var, promoted_var).replace(memref_type, promoted_type)
        
        # Create copy operations from original to promoted and back
        # Copy from original to promoted at the beginning
        copy_to_promoted = f"""
  // Copy data to promoted memory space
  affine.for %i0 = 0 to 10 {{
    %val = affine.load {memref_var}[%i0] : {memref_type}
    affine.store %val, {promoted_var}[%i0] : {promoted_type}
  }}
"""
        
        # Use the promoted variable for all operations
        # Find all uses of the original variable and replace with promoted variable
        text_str = final_text.decode("utf-8")
        operations_with_original = text_str.count(memref_var)
        
        # Copy back from promoted to original at the end
        copy_back = f"""
  // Copy data back from promoted memory space
  affine.for %i0 = 0 to 10 {{
    %val = affine.load {promoted_var}[%i0] : {promoted_type}
    affine.store %val, {memref_var}[%i0] : {memref_type}
  }}
"""
        
        # Only proceed if we found usages of the original variable
        if operations_with_original <= 1:  # Just the allocation itself
            return tree
            
        # Insert the promoted allocation after the original allocation
        new_code = alloc_line + "\n  " + promoted_alloc + copy_to_promoted
        
        # Find a suitable place to insert the copy-back operation
        # Ideally before deallocation or at the end of the function
        if "memref.dealloc" in text_str and memref_var in text_str.split("memref.dealloc")[1]:
            # Insert before deallocation
            dealloc_index = text_str.find("memref.dealloc", text_str.find(memref_var))
            if dealloc_index != -1:
                prefix = text_str[:dealloc_index]
                suffix = text_str[dealloc_index:]
                new_text = prefix + copy_back + "  " + suffix
            else:
                # If no deallocation found, append at end
                new_text = text_str + copy_back
        else:
            # Append at the end of the function
            new_text = text_str + copy_back
        
        # Replace the original allocation with the new code
        new_text = new_text.replace(alloc_line, new_code)
        
        # Add deallocation for the promoted buffer
        dealloc_promoted = f"  memref.dealloc {promoted_var} : {promoted_type}\n"
        if "memref.dealloc" in new_text and memref_var in new_text.split("memref.dealloc")[1]:
            dealloc_index = new_text.find("memref.dealloc", new_text.find(memref_var))
            if dealloc_index != -1:
                new_text = new_text[:dealloc_index] + dealloc_promoted + new_text[dealloc_index:]
        else:
            new_text = new_text + dealloc_promoted
        
        # Parse and return the new tree
        return mlir_parser.parse(new_text.encode("utf-8")) 