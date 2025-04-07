'use strict';

module.exports = {
  gpu_dialect: $ => prec.right(choice(
    // GPU Launch operation
    seq('gpu.launch',
      field('asyncDependencies', optional(seq('[', $._value_use_list, ']'))),
      field('clusters', optional(seq('clusters', '(', $._value_use_list, ')', 'in', $._value_assignment_list))),
      field('blocks', seq('blocks', '(', $._value_use_list, ')', 'in', $._value_assignment_list)),
      field('threads', seq('threads', '(', $._value_use_list, ')', 'in', $._value_assignment_list)),
      field('dynamicSharedMemorySize', optional(seq('dynamic_shared_memory_size', $.value_use))),
      field('memoryAttribution', optional($._memory_attribution)),
      field('body', $.region),
      field('attributes', optional($.attribute))),

    // GPU Alloc operation
    seq('gpu.alloc',
      field('asyncDependencies', optional(seq('[', $._value_use_list, ']'))),
      field('hostShared', optional('host_shared')),
      field('dynamicSizes', $._value_use_list_parens),
      field('symbolOperands', optional(seq('[', $._value_use_list, ']'))),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // GPU Dealloc operation
    seq('gpu.dealloc',
      field('asyncDependencies', optional(seq('[', $._value_use_list, ']'))),
      field('memref', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // GPU Memcpy operation
    seq('gpu.memcpy',
      field('asyncDependencies', optional(seq('[', $._value_use_list, ']'))),
      field('dst', $.value_use), ',',
      field('src', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // GPU Wait operation
    seq('gpu.wait',
      field('asyncDependencies', optional(seq('[', $._value_use_list, ']'))),
      field('attributes', optional($.attribute)),
      field('return', optional($._type_annotation))),

    // GPU Barrier operation
    seq('gpu.barrier',
      field('attributes', optional($.attribute))),

    // GPU Thread ID operations
    seq(choice('gpu.thread_id', 'gpu.block_id', 'gpu.block_dim', 'gpu.grid_dim'),
      field('dimension', $.string_literal),
      field('upperBound', optional(seq('upper_bound', $.value_use))),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // GPU Shuffle operation
    seq('gpu.shuffle',
      field('mode', $.string_literal),
      field('value', $.value_use), ',',
      field('offset', $.value_use), ',',
      field('width', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // GPU Sync operation
    seq('gpu.sync',
      field('attributes', optional($.attribute))),

    // GPU Yield operation
    seq('gpu.yield',
      field('values', optional($._value_use_type_list)),
      field('attributes', optional($.attribute))),

    // GPU Return operation
    seq('gpu.return',
      field('operands', optional($._value_use_list)),
      field('attributes', optional($.attribute))),

    // GPU Module operation
    seq('gpu.module',
      field('symName', $.string_literal),
      field('offloadingHandler', optional(seq('<', $.attribute, '>'))),
      field('targets', optional($._attribute_list)),
      field('attributes', optional($.attribute)),
      field('body', $.region)),

    // GPU Function operation
    seq('gpu.func',
      field('symName', $.string_literal),
      field('arguments', $._argument_list),
      field('memoryAttribution', optional($._memory_attribution)),
      field('kernel', optional('kernel')),
      field('attributes', optional($.attribute)),
      field('body', $.region)),

    // GPU Launch Function operation
    seq('gpu.launch_func',
      field('asyncDependencies', optional(seq('[', $._value_use_list, ']'))),
      field('asyncObject', optional(seq('<', $.value_use, ':', $._type_annotation, '>'))),
      field('kernel', $.string_literal),
      field('clusters', optional(seq('clusters', 'in', '(', $._value_use_list, ')'))),
      field('blocks', seq('blocks', 'in', '(', $._value_use_list, ')')),
      field('threads', seq('threads', 'in', '(', $._value_use_list, ')')),
      field('dynamicSharedMemorySize', optional(seq('dynamic_shared_memory_size', $.value_use))),
      field('kernelOperands', optional($._value_use_type_list)),
      field('attributes', optional($.attribute)),
      field('return', optional($._type_annotation)))
  ))
}
