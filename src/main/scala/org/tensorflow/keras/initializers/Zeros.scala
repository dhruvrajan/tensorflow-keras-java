package org.tensorflow.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.types.TInt32
import org.tensorflow.types.family.TNumber

class Zeros extends Initializer {
  override def initialize[T <: TNumber](tf: Ops, shape: Operand[TInt32], dtype: Class[T]): Operand[T] =
    tf.zeros(shape, dtype)
}
