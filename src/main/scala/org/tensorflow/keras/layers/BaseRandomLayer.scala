package org.tensorflow.keras.layers

import org.tensorflow.keras.utils.Backend
import org.tensorflow.types.family.TNumber

/** A layer to handle random number creation and save-model behavior.
  *
  * @param seed optional integer, used to create RandomGenerator.
  * @param forceGenerator  default to `false`, whether to force the
  *     RandomGenerator to use the code branch of tf.random.Generator.
  */
abstract class BaseRandomLayer[T <: TNumber](seed: Option[Long] = None, forceGenerator: Boolean = false,
                                             numInputs: Int)
  extends Layer[T](numInputs) {

  protected val randomGenerator = new Backend.RandomGenerator(seed, forceGenerator = forceGenerator)
}
