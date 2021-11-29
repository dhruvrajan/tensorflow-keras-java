package org.tensorflow.keras.initializers

import org.tensorflow.keras.initializers.VarianceScaling.{FanAvg, Uniform}

/** The Glorot uniform initializer, also called Xavier uniform initializer. */
class GlorotUniform(seed: Option[Long] = None)
  extends VarianceScaling(scale = 1.0, mode = FanAvg, distribution = Uniform, seed = seed)