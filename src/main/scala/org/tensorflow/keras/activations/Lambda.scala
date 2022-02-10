package org.tensorflow.keras.activations

import org.tensorflow.Operand
import org.tensorflow.keras.mixin.ActivationFunction
import org.tensorflow.op.Ops
import org.tensorflow.types.family.TNumber

/**
  * Creates an `Activation` from an unnamed function.
  *
  * @param unnamedActivation An activation function.
  * @param < T>
  */
class Lambda[T <: TNumber](val activation: ActivationFunction[T]) extends Activation[T] {
  /**
    * Applies the given activation.
    */
  override protected def callOne(tf: Ops, inputs: Operand[T]): Operand[T] = activation.apply(tf, inputs)
}
