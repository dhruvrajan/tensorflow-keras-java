package org.tensorflow.keras.activations

import org.tensorflow.Operand
import org.tensorflow.keras.layers.Layer
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.family.TNumber

/**
  * Base activation function class.
  */
abstract class Activation[T <: TNumber]() extends Layer[T](1) {
  override def build(tf: Ops, inputShape: Shape): Unit = {
    // Activations don't need state to be built and added to a graph. Does nothing.
  }

  override def computeOutputShape(inputShape: Shape): Shape = { // Activation functions should not change the shape of the input.
    inputShape
  }

  @SafeVarargs final def call(tf: Ops, inputs: Operand[T]*): Operand[T] = callOne(tf, inputs(0))

  /**
    * Calls the activation function. Override this when defining an activation function.
    */
  protected def callOne(tf: Ops, inputs: Operand[T]): Operand[T]
}
