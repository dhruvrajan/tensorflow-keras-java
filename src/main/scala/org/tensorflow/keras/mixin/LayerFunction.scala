package org.tensorflow.keras.mixin

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.types.family.TType

@FunctionalInterface trait LayerFunction[T <: TType] {
  @SuppressWarnings(Array("unchecked")) def apply(tf: Ops, inputs: Operand[T]*): Operand[T]
}
