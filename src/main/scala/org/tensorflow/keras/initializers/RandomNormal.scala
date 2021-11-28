package org.tensorflow.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.types.TInt32
import org.tensorflow.types.family.TNumber

class RandomNormal(val mean: Float, val stdev: Float, val p1: Float, val p2: Float) extends Initializer {
  override def initialize[T <: TNumber](tf: Ops, shape: Operand[TInt32], dtype: Class[T]): Operand[T] =
    tf.random.parameterizedTruncatedNormal(
      shape,
      tf.dtypes.cast(tf.constant(this.mean  ), dtype),
      tf.dtypes.cast(tf.constant(this.stdev ), dtype),
      tf.dtypes.cast(tf.constant(this.p1    ), dtype),
      tf.dtypes.cast(tf.constant(this.p2    ), dtype)
    )
}
