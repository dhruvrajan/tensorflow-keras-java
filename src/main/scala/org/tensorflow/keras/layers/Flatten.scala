package org.tensorflow.keras.layers

import org.tensorflow.Operand
import org.tensorflow.keras.utils.TensorShape
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import org.tensorflow.types.TInt32
import org.tensorflow.types.family.TNumber


object Flatten {
  private val FLATTEN_INPUT_LENGTH = 1
}

class Flatten[T <: TNumber]() extends Layer[T](Flatten.FLATTEN_INPUT_LENGTH) {
  private var units: Constant[TInt32] = null

  override def build(tf: Ops, inputShape: Shape): Unit = {
    val tensorShape = new TensorShape(inputShape)
    this.units = tf.constant(Array[Int](-1, (tensorShape.numElements / Math.abs(tensorShape.size(0))).toInt))
  }

  override def computeOutputShape(inputShape: Shape): Shape = { // leaves unknown dimensions unknown
    Shape.of(new TensorShape(inputShape).numElements)
  }

  @SafeVarargs final def call(tf: Ops, inputs: Operand[T]*): Operand[T] = this.callOne(tf, inputs(0))

  private def callOne(tf: Ops, input: Operand[T]) = tf.reshape(input, this.units)
}
