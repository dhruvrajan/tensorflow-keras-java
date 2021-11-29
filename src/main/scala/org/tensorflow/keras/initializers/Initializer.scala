package org.tensorflow.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.keras.utils.Keras
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign
import org.tensorflow.types.TInt32
import org.tensorflow.types.family.TNumber

object Initializer {
  /** Computes the number of input and output units for a weight shape.
    *
    * @param shape  Integer shape tuple or TF tensor shape.
    * @return       A tuple of integer scalars (fan_in, fan_out).
    */
  def computeFans(shape: Operand[TInt32]): (Int, Int) = {
    shape.rank() match {
      case i if i < 1 => (1, 1)
      case 1 =>
//        val fan = shape(0)
//        (fan, fan)
        ???
      case 2 =>
//        (shape(0), shape(1))
        ???
      case _ =>
        // Assuming convolution kernels (2D, 3D, or more).
        // kernel shape: (..., input_depth, depth)
        var receptiveFieldSize = 1
//        for (dim <- shape(/*:- 2*/)) {
//          receptiveFieldSize *= dim
//        }
//        (
//          shape(-2) * receptiveFieldSize,
//          shape(-1) * receptiveFieldSize
//        )
        ???
    }
  // return int(fan_in), int(fan_out)
  }
}
abstract class Initializer {
  /**
    * Adds an `Assign` Op to the graph to initialize
    * a tensorflow variable as specified by the initializer.
    *
    * @param tf Tensorflow Ops Accessor
    * @param in Variable to initialize
    * @return Assign Operand created
    */
  def apply[T <: TNumber](tf: Ops, in: Operand[T], dtype: Class[T]): Assign[T] =
    tf.assign(in, this.initialize(tf, Keras.shapeOperand(tf, in.asOutput.shape), dtype))

  /**
    * Returns a Tensor object initialized as
    * specified by the initializer.
    *
    * @param tf    Tensorflow Ops Handle
    * @param shape Shape of the tensor
    */
  def initialize[T <: TNumber](tf: Ops, shape: Operand[TInt32], dtype: Class[T]): Operand[T]
}
