from parser_utils import mlir_parser

from tree_sitter import Node, Tree
from mutation import Mutator
import random


class AffineDataCopyGenerate(Mutator):
    def mutate(self, tree: Tree):
        """
        Generate explicit data copying for affine memory operations to simulate
        the affine-data-copy-generate pass which copies data from slow memory space
        to fast memory space for performance improvement.
        """
        final_text = tree.text

        # Configuration parameters (similar to the pass options)
        fast_memory_space = random.randint(1, 4)  # Default is 1 in the pass
        slow_memory_space = 0  # Default is 0 in the pass
        generate_dma = random.choice([True, False])  # Randomly choose between DMA and point-wise copy
        
        # Find affine load/store operations which are candidates for memory copying
        affine_memory_ops = []
        
        def _find_affine_memory_ops(node: Node):
            # Look for affine.load and affine.store operations
            if node.type == "affine_dialect":
                text = node.text.decode("utf-8")
                if "affine.load" in text or "affine.store" in text:
                    affine_memory_ops.append(node)
            
            for child in node.children:
                _find_affine_memory_ops(child)
        
        _find_affine_memory_ops(tree.root_node)
        
        # If no affine memory operations found, return the original tree
        if not affine_memory_ops:
            print("No affine memory operations found, cannot apply affine-data-copy-generate")
            return tree
        
        # Select a random affine memory operation to transform
        target_op = random.choice(affine_memory_ops)
        target_text = target_op.text.decode("utf-8")
        
        # Extract operation type and memory reference
        is_load = "affine.load" in target_text
        operation_type = "affine.load" if is_load else "affine.store"
        
        # Parse the operation to extract key components
        lines = target_text.strip().split('\n')
        op_line = next((line for line in lines if operation_type in line), "")
        
        if not op_line:
            return tree  # Failed to find the operation line
        
        # Extract the memref and index expressions
        try:
            if is_load:
                # Format: %result = affine.load %memref[index_expressions] : memref_type
                parts = op_line.split('=')
                result_var = parts[0].strip()
                op_parts = parts[1].strip().split(':')
                memref_part = op_parts[0].strip().split('[')[0].strip().split()[-1]
                memref_type = op_parts[1].strip()
                
                # Extract index expressions
                if '[' in op_parts[0] and ']' in op_parts[0]:
                    index_expr = op_parts[0].split('[')[1].split(']')[0].strip()
                else:
                    index_expr = ""
            else:
                # Format: affine.store %value, %memref[index_expressions] : memref_type
                op_parts = op_line.split(':')
                mem_parts = op_parts[0].strip().split(',')
                value_var = mem_parts[0].split()[-1]
                memref_part = mem_parts[1].strip().split('[')[0].strip()
                memref_type = op_parts[1].strip()
                
                # Extract index expressions
                if '[' in mem_parts[1] and ']' in mem_parts[1]:
                    index_expr = mem_parts[1].split('[')[1].split(']')[0].strip()
                else:
                    index_expr = ""
        except:
            # If parsing fails, return the original tree
            print("Failed to parse the affine memory operation")
            return tree
        
        # Generate a new temporary buffer in the fast memory space
        # For simplicity, we'll assume the original memref type and modify its memory space
        new_memref_type = memref_type.replace("memref<", f"memref<").replace(">", f", {fast_memory_space}>")
        if "memref<" not in new_memref_type:
            # Default to a simple memref type if parsing fails
            new_memref_type = f"memref<10xi32, {fast_memory_space}>"
        
        # Generate a unique name for the new buffer
        temp_buffer = f"%temp_buffer_{random.randint(1000, 9999)}"
        
        # Create the buffer allocation
        alloc_code = f"  {temp_buffer} = memref.alloc() : {new_memref_type}\n"
        
        # Create the copy operations
        if generate_dma:
            # Using DMA operations
            if is_load:
                # Copy from slow to fast before load
                copy_from_slow = f"  affine.dma_start {memref_part}[{index_expr}], {temp_buffer}[{index_expr}], %c1, %c0 : {memref_type}, {new_memref_type}, memref<1xi32>\n"
                copy_to_slow = ""  # No need to copy back for load
                
                # Replace the original load with load from fast memory
                new_load = f"  {result_var} = affine.load {temp_buffer}[{index_expr}] : {new_memref_type}\n"
                new_op_text = copy_from_slow + new_load
            else:
                # For store, first store to fast memory, then copy back to slow
                new_store = f"  affine.store {value_var}, {temp_buffer}[{index_expr}] : {new_memref_type}\n"
                copy_to_slow = f"  affine.dma_start {temp_buffer}[{index_expr}], {memref_part}[{index_expr}], %c1, %c0 : {new_memref_type}, {memref_type}, memref<1xi32>\n"
                new_op_text = new_store + copy_to_slow
        else:
            # Using simple load/store for copying
            if is_load:
                # Copy from slow to fast before load
                copy_from_slow = f"  affine.for %i0 = 0 to 1 {{\n"
                copy_from_slow += f"    %temp_val = affine.load {memref_part}[{index_expr}] : {memref_type}\n"
                copy_from_slow += f"    affine.store %temp_val, {temp_buffer}[{index_expr}] : {new_memref_type}\n"
                copy_from_slow += f"  }}\n"
                
                # Replace the original load with load from fast memory
                new_load = f"  {result_var} = affine.load {temp_buffer}[{index_expr}] : {new_memref_type}\n"
                new_op_text = copy_from_slow + new_load
            else:
                # For store, first store to fast memory, then copy back to slow
                new_store = f"  affine.store {value_var}, {temp_buffer}[{index_expr}] : {new_memref_type}\n"
                copy_to_slow = f"  affine.for %i0 = 0 to 1 {{\n"
                copy_to_slow += f"    %temp_val = affine.load {temp_buffer}[{index_expr}] : {new_memref_type}\n"
                copy_to_slow += f"    affine.store %temp_val, {memref_part}[{index_expr}] : {memref_type}\n"
                copy_to_slow += f"  }}\n"
                new_op_text = new_store + copy_to_slow
        
        # Add memory deallocation
        dealloc_code = f"  memref.dealloc {temp_buffer} : {new_memref_type}\n"
        
        # Create constants needed for DMA operations (if using DMA)
        constants_code = ""
        if generate_dma:
            constants_code = "  %c0 = arith.constant 0 : index\n  %c1 = arith.constant 1 : index\n"
        
        # Combine all the generated code
        full_code = constants_code + alloc_code + new_op_text + dealloc_code
        
        # Replace the original operation with our generated code
        text_str = final_text.decode("utf-8")
        replaced_text = text_str.replace(target_text.strip(), full_code.strip())
        
        # If replacement failed, try a simpler approach
        if replaced_text == text_str:
            replaced_text = text_str.replace(op_line, full_code)
        
        if replaced_text != text_str:
            return mlir_parser.parse(replaced_text.encode("utf-8"))
        else:
            print("Failed to replace the affine memory operation")
            return tree 