package org.tensorflow.keras.layers

import org.tensorflow.Operand
import org.tensorflow.keras.utils.{Backend, ControlFlowUtil}
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.family.{TNumber, TType}

/** Applies Dropout to the input.
  * The Dropout layer randomly sets input units to 0 with a frequency of `rate`
  * at each step during training time, which helps prevent over-fitting.
  * Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over
  * all inputs is unchanged.
  * Note that the Dropout layer only applies when `training` is set to True
  * such that no values are dropped during inference. When using `model.fit`,
  * `training` will be appropriately set to True automatically, and in other
  * contexts, you can set the kwarg explicitly to True when calling the layer.
  * (This is in contrast to setting `trainable=False` for a Dropout layer.
  * `trainable` does not affect the layer's behavior, as Dropout does
  * not have any variables/weights that can be frozen during training.)
  * {{{
  * >>> tf.random.set_seed(0)
  * >>> layer = tf.keras.layers.Dropout(.2, input_shape=(2,))
  * >>> data = np.arange(10).reshape(5, 2).astype(np.float32)
  * >>> print(data)
  * [[0. 1.]
  *  [2. 3.]
  *  [4. 5.]
  *  [6. 7.]
  *  [8. 9.]]
  * >>> outputs = layer(data, training=True)
  * >>> print(outputs)
  * tf.Tensor(
  * [[ 0.    1.25]
  *  [ 2.5   3.75]
  *  [ 5.    6.25]
  *  [ 7.5   8.75]
  *  [10.    0.  ]], shape=(5, 2), dtype=float32)
  *  }}}
  * Args:
  *   rate: Float between 0 and 1. Fraction of the input units to drop.
  *   noise_shape: 1D integer tensor representing the shape of the
  *     binary dropout mask that will be multiplied with the input.
  *     For instance, if your inputs have shape
  *     `(batch_size, timesteps, features)` and
  *     you want the dropout mask to be the same for all timesteps,
  *     you can use `noise_shape=(batch_size, 1, features)`.
  *   seed: A Python integer to use as random seed.
  * Call arguments:
  *   inputs: Input tensor (of any rank).
  *   training: Python boolean indicating whether the layer should behave in
  *     training mode (adding dropout) or in inference mode (doing nothing).
  */
class Dropout[T <: TNumber](rate: Float, noiseShape: Shape = Shape.unknown(), seed: Option[Long] = None)
  extends BaseRandomLayer[T](seed = seed, numInputs = 1) {

  if (!(0f <= rate && rate <= 1f))
    throw new IllegalArgumentException(s"Invalid value $rate received for `rate`, expected a value between 0 and 1.")

//  this.supports_masking = True

  override protected def build(tf: Ops, inputShape: Shape): Unit =
    randomGenerator.maybeInit(tf)  // pylint: disable=protected-access

  private def getNoiseShape(tf: Ops, inputs: Operand[T]): Shape = {
    // Subclasses of `Dropout` may implement `_get_noise_shape(this, inputs)`,
    // which will override `this.noise_shape`, and allows for custom noise
    // shapes with dynamically sized inputs.
    if (noiseShape.isUnknown) return Shape.unknown() // None

    val concreteInputsShape = tf.shape(inputs).shape()
    var res = Shape.scalar()
    for (i <- 0 until noiseShape.numDimensions()) {
      val value = noiseShape.size(i)
      res = res.append(if (value == Shape.UNKNOWN_SIZE) concreteInputsShape.size(i) else value)
    }
    res // tf.convert_to_tensor(res)
  }

  override protected def call(tf: Ops, inputs: Seq[Operand[T]], training: Option[Boolean]): Operand[T] =
    callOne(tf, inputs.head)

  private def callOne(tf: Ops, inputs: Operand[T]): Operand[T] = {
    val training = false // XXX TODO
    if (!training)
      ??? // training = Backend.learningPhase()

    def droppedInputs() = ???
//      randomGenerator.dropout(
//        inputs, rate, noiseShape = getNoiseShape(tf, inputs))

    val output = tf.identity(inputs)

    // XXX TODO
//    val output = ControlFlowUtil.smartCond(tf, training, droppedInputs,
//      lambda: tf.identity(inputs)
//    )
    output
  }

  override def computeOutputShape(inputShape: Shape): Shape = inputShape
}

