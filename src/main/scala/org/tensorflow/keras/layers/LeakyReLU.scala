package org.tensorflow.keras.layers
import org.tensorflow.Operand
import org.tensorflow.keras.utils.Backend
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.family.TNumber

/** Leaky version of a Rectified Linear Unit.
  * It allows a small gradient when the unit is not active:
  * {{{
  *   f(x) = alpha * x if x < 0
  *   f(x) = x if x >= 0
  * }}}
  * Usage:
  * >>> layer = tf.keras.layers.LeakyReLU()
  * >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  * >>> list(output.numpy())
  * [-0.9, -0.3, 0.0, 2.0]
  * >>> layer = tf.keras.layers.LeakyReLU(alpha=0.1)
  * >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  * >>> list(output.numpy())
  * [-0.3, -0.1, 0.0, 2.0]
  *
  * Input shape:
  *   Arbitrary. Use the keyword argument `input_shape`
  *   (tuple of integers, does not include the batch axis)
  *   when using this layer as the first layer in a model.
  * Output shape:
  *   Same shape as the input.
  *
  * @param alpha a float >= 0. Negative slope coefficient. Defaults to 0.3.
  */
class LeakyReLU[T <: TNumber](alpha: Float = 0.3f) extends Layer[T](1) {
  // this.supports_masking = true

  /**
    * Defines the layer's logic, in terms of input operands, and variables.
    *
    * @param tf     Tensorflow Ops accessor.
    * @param inputs A sequence of TF Operands
    * @return The transformed input tensors, according to the layer's logic.
    */
  override protected def call(tf: Ops, inputs: Operand[T]*): Operand[T] =
    call(tf, inputs(0))

  private def call(tf: Ops, inputs: Operand[T]): Operand[T] =
    Backend.relu(tf, inputs, alpha = alpha)

  override protected def build(tf: Ops, inputShape: Shape): Unit =
    built = true

  override def computeOutputShape(inputShape: Shape): Shape = inputShape
}
