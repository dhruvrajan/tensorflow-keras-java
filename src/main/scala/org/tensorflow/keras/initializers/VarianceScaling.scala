package org.tensorflow.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.keras.initializers.VarianceScaling.{Distribution, FanIn, FanOut, Mode, TruncatedNormal, UntruncatedNormal}
import org.tensorflow.op.Ops
import org.tensorflow.types.TInt32
import org.tensorflow.types.family.TNumber

object VarianceScaling {
  sealed trait Mode
  case object FanIn  extends Mode
  case object FanOut extends Mode
  case object FanAvg extends Mode
  
  sealed trait Distribution
  case object Uniform           extends Distribution
  case object TruncatedNormal   extends Distribution
  case object UntruncatedNormal extends Distribution
}
/** Initializer capable of adapting its scale to the shape of weights tensors. */
abstract class VarianceScaling(scale          : Double                  = 1.0,
                               mode           : Mode                    = FanIn,
                               distribution   : Distribution            = TruncatedNormal,
                               seed           : Option[Long]            = None,
                               partitionShape : Option[Operand[TInt32]] = None
                              )
  extends Initializer {

  if (scale <= 0.0) throw new IllegalArgumentException(s"`scale` ($scale) must be positive float.")

  override def initialize[T <: TNumber](tf: Ops, shape0: Operand[TInt32], dtype: Class[T]): Operand[T] = {
    // dtype = _assert_float_dtype(_get_dtype(dtype))
    var scale = this.scale
    val (fanIn, fanOut) = Initializer.computeFans(shape0)
    val shape = partitionShape.getOrElse(shape0)
    if (mode == FanIn) {
      scale /= math.max(1.0, fanIn)
    } else if (mode == FanOut) {
      scale /= math.max(1.0, fanOut)
    } else {
      scale /= math.max(1.0, (fanIn + fanOut) / 2.0)
    }
    val rng = tf.random

    if (this.distribution == TruncatedNormal) {
      // constant from scipy.stats.truncnorm.std(a =- 2, b = 2, loc = 0., scale = 1.)
      val stdDev = math.sqrt(scale) / 0.87962566103423978
      // rng.truncatedNormal[T](shape, 0.0, stdDev, dtype)
      rng.parameterizedTruncatedNormal[T](shape,
        tf.dtypes.cast(tf.constant(0.0), dtype),
        tf.dtypes.cast(tf.constant(stdDev), dtype),
        tf.dtypes.cast(???, dtype),
        tf.dtypes.cast(???, dtype),
      )
    } else if (distribution == UntruncatedNormal) {
      val stdDev = math.sqrt(scale)
      // rng.random_normal(shape, 0.0, stdDev, dtype)
      // rng.randomStandardNormal()
      ???
    } else {
      val limit = math.sqrt(3.0 * scale)
      // rng.random_uniform(shape, -limit, limit, dtype)
      // rng.randomUniform()
      ???
    }
  }
}
