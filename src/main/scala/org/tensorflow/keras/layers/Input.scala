package org.tensorflow.keras.layers

import org.tensorflow.Operand
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.types.family.TNumber

class Input[T <: TNumber](units: Shape, batchSize: Long = Shape.UNKNOWN_SIZE)
  extends Layer[T](0) {

  private var input: Placeholder[T] = null

  override def build(tf: Ops, inputShape: Shape): Unit =
    throw new UnsupportedOperationException(
      "Cannot build an input layer with an input shape; it doesn't take any inputs. Use Input.build(Ops tf, Class<T> dtype)"
    )

  override def computeOutputShape(inputShape: Shape): Shape =
    throw new UnsupportedOperationException("Cannot call computeOutputShape on")

  def computeOutputShape: Shape = input.asOutput.shape

  def build(tf: Ops, dtype: Class[T]): Unit = {
    this.dtype = dtype
    // System.out.println(dtype);
    this.input = tf.placeholder(dtype, Placeholder.shape(units.prepend(batchSize)))
    this.built = true
  }

  override final def call(tf: Ops, inputs: Seq[Operand[T]], training: Option[Boolean]): Operand[T] = input
}