package org.tensorflow.keras.layers

import org.tensorflow.Operand
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.family.TNumber
import org.tensorflow.utils.TensorShape

/** Layer that reshapes inputs into the given shape. */
class Reshape[T <: TNumber](targetShape: Shape) extends Layer[T](1) {
  override protected def build(tf: Ops, inputShape: Shape): Unit = ???

  override def computeOutputShape(inputShape0: Shape): Shape = {
//    val inputShape  = new TensorShape(inputShape0).as_list()
//    var outputShape = (inputShape(0))
//    if (None in inputShape(1:)) {
//      // input shape (partially) unknown ? replace -1's with None's
//      val FOO = s if s != -1 else None for s in targetShape
//      outputShape += tuple(FOO)
//    } else {
//      var output_shape = (inputShape(0))
//      output_shape += fixUnknownDimension(inputShape(1:), targetShape)
//    }
//    new TensorShape(outputShape).toShape
    ???
  }

  private def fixUnknownDimension(input_shape: Shape, output_shape: Shape): Shape = ???

  override protected def call(tf: Ops, inputs: Operand[T]*): Operand[T] =
    callOne(tf, inputs(0))

  private def callOne(tf: Ops, inputs: Operand[T]): Operand[T] = {
//    val result = tf.reshape(inputs, (tf.shape(inputs)(0),) + targetShape)
//    if (!tf.scope().env().isEager) {
//      // Set the static shape for the result since it might lost during array_ops
//      // reshape, eg, some `None` dim in the result could be inferred.
//      result.set_shape(computeOutputShape(inputs.shape))
//    }
//    result
    ???
  }
}
