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
      field('memoryAttribution', optional($.attribute)),
      field('body', $.region),
      field('attributes', optional($.attribute))),

    // GPU Alloc operation - 修改支持host_shared
    // operation ::= `gpu.alloc` custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) (` ` `host_shared` $hostShared^)? ` `
    // `(` $dynamicSizes `)` (`` `[` $symbolOperands^ `]`)? attr-dict `:` type($memref)
    seq('gpu.alloc',
      field('async', optional('async')),
      field('asyncDependencies', optional(seq('[', $._value_use_list, ']'))),
      field('hostShared', optional('host_shared')),
      field('dynamicSizes', optional($._value_use_list_parens)), // 使dynamicSizes可选
      field('symbolOperands', optional(seq('[', $._value_use_list, ']'))),
      field('attributes', optional($.attribute)),
      field('return', optional($._type_annotation))),

    // GPU Dealloc operation - 添加async支持
    seq('gpu.dealloc',
      field('async', optional('async')),
      field('asyncDependencies', optional(seq('[', $._value_use_list, ']'))),
      field('memref', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', optional($._type_annotation))),

    // GPU Memcpy operation
    seq('gpu.memcpy',
      field('asyncDependencies', optional(seq('[', $._value_use_list, ']'))),
      field('dst', $.value_use), ',',
      field('src', $.value_use),
      field('attributes', optional($.attribute)),
      field('return', $._type_annotation)),

    // GPU Wait operation - 添加async支持
    seq('gpu.wait',
      field('async', optional('async')),
      field('asyncDependencies', optional(seq('[', $._value_use_list, ']'))),
      field('attributes', optional($.attribute)),
      field('return', optional($._type_annotation))),

    // GPU Barrier operation
    seq('gpu.barrier',
      field('attributes', optional($.attribute))),

    // GPU Thread ID operations - 支持block_id x/y/z简写形式
    seq(choice('gpu.thread_id', 'gpu.block_id', 'gpu.block_dim', 'gpu.grid_dim'),
      field('dimension', choice($.string_literal, token('x'), token('y'), token('z'))),
      field('upperBound', optional(seq('upper_bound', $.value_use))),
      field('attributes', optional($.attribute)),
      field('return', optional($._type_annotation))),

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
      field('operands', optional($._value_use_type_list)),
      field('attributes', optional($.attribute))),

    // GPU Module operation
    seq('gpu.module',
      field('symName', $.symbol_ref_id), // 改为symbol_ref_id而不是string_literal
      field('offloadingHandler', optional(seq('<', $.attribute, '>'))),
      field('targets', optional($.attribute)),
      field('attributes', optional($.attribute)),
      field('body', $.region)),

    // GPU Function operation
    // op ::= `gpu.func` symbol-ref-id `(` argument-list `)` (`->`
// function-result-list)?
// memory-attribution `kernel`? function-attributes? region
// memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
                // (`private` `(` ssa-id-and-type-list `)`)?
    seq('gpu.func',
      field('symName', $.symbol_ref_id), // 改为symbol_ref_id
      field('arguments', $._value_use_type_list),
      field('memoryAttribution', optional($.attribute)),
      field('kernel', optional('kernel')),
      field('attributes', optional($.attribute)),
      field('body', $.region)),

    // GPU Launch Function operation - 添加async支持
    seq('gpu.launch_func',
      field('async', optional('async')),
      field('asyncDependencies', optional(seq('[', $._value_use_list, ']'))),
      field('asyncObject', optional(seq('<', $.value_use, ':', $._type_annotation, '>'))),
      field('kernel', choice($.symbol_ref_id, $.string_literal)), // 支持符号引用
      field('clusters', optional(seq('clusters', 'in', '(', $._value_use_list, ')'))),
      field('blocks', seq('blocks', 'in', '(', $._value_use_list, ')')),
      field('threads', seq('threads', 'in', '(', $._value_use_list, ')')),
      field('dynamicSharedMemorySize', optional(seq('dynamic_shared_memory_size', $.value_use))),
      field('kernelOperands', optional(seq('args', '(', optional($._value_use_list), ')'))), // 修改为args语法
      field('attributes', optional($.attribute)),
      field('return', optional($._type_annotation)))
  ))
}