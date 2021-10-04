package org.tensorflow.keras.utils

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.nn.LeakyRelu
import org.tensorflow.types.family.TNumber

object Backend {
  /** Rectified linear unit.
    * With default values, it returns element-wise `max(x, 0)`.
    * Otherwise, it follows:
    * `f(x) = max_value` for `x >= max_value`,
    * `f(x) = x` for `threshold <= x < max_value`,
    * `f(x) = alpha * (x - threshold)` otherwise.
    *
    * @param alpha      A scalar, slope of negative section (default=`0.`).
    * @param maxValue   Saturation threshold.
    * @param threshold  Threshold value for activation with threshold.
    */
  def relu[T <: TNumber](tf: Ops, x: Operand[T], alpha: Float = 0.0f, maxValue: Option[Float] = None,
                         threshold: Float = 0.0f): Operand[T] = {
    var xv = x
    // While x can be a tensor or variable, we also see cases where
    // numpy arrays, lists, tuples are passed as well.
    // lists, tuples do not have 'dtype' attribute.
    val dtype = xv.`type`() // getattr(xv, "dtype", floatx())
    var negativePart: Operand[T] = null
    if (alpha != 0.0) {
      if (maxValue.isEmpty && threshold == 0.0) {
        return tf.nn.leakyRelu(xv, LeakyRelu.alpha(alpha))
      }

      negativePart = if (threshold != 0.0) {
        val thresholdT = tf.dtypes.cast(tf.constant(threshold), dtype)
        tf.nn.relu(tf.math.add(tf.math.neg(xv), thresholdT)) // -xv + threshold XXX TODO correct?
      } else {
        tf.nn.relu(tf.math.neg(xv)) // -xv XXX TODO correct?
      }
    }
    var clipMax = maxValue.isDefined

    if (threshold != 0.0) {
      // computes x for x > threshold else 0
      val thresholdT = tf.dtypes.cast(tf.constant(threshold), dtype)
      xv = tf.math.mul(xv,
        tf.dtypes.cast(tf.math.greater(xv, thresholdT), dtype)
      ) // x * tf.cast(tf.greater(x, threshold), dtype=dtype)
    } else if (maxValue.contains(6f)) {
      // if no threshold, then can use nn.relu6 native TF op for performance
      xv = tf.nn.relu6(xv)
      clipMax = false
    } else {
      xv = tf.nn.relu(xv)
    }

    if (clipMax) {
//      val maxValueT = constantToTensor(maxValue.get, xv.dtype.base_dtype)
      val maxValueT = tf.dtypes.cast(tf.constant(maxValue.get), dtype)
//      val zero      = constantToTensor(0, xv.dtype.base_dtype)
      val zero      = tf.dtypes.cast(tf.constant(0f), dtype)
      xv = tf.clipByValue(xv, zero, maxValueT)
    }

    if (alpha != 0.0) {
      // val alphaT = toTensor(alpha, xv.dtype.base_dtype)
      val alphaT = tf.dtypes.cast(tf.constant(alpha), dtype)
      xv = tf.math.sub(x, tf.math.mul(alphaT, negativePart))
    }
    xv
  }

//  /** Convert the input `x` to a tensor of type `dtype`.
//    * This is slightly faster than the _to_tensor function, at the cost of
//    * handling fewer cases.
//    *
//    * @param x      An object to be converted (numpy arrays, floats, ints and lists of them).
//    * @param dtype  The destination type.
//    */
//  private def constantToTensor(tf: Ops, x: Float, dtype) =
//    tf.constant(x, dtype = dtype)
//
//  /** Convert the input `x` to a tensor of type `dtype`.
//    *
//    * @param x      An object to be converted (numpy array, list, tensors).
//    * @param dtype  The destination type.
//    */
//  private def toTensor(tf: Ops, x: Float, dtype) =
//    tf.convert_to_tensor(x, dtype = dtype)

  class RandomGenerator(seed: Option[Long] = None, forceGenerator: Boolean = false) {
    private var built = false

    /** Lazily init the RandomGenerator.
      * The TF API executing_eagerly_outside_functions() has some side effect, and
      * couldn't be used before API like tf.enable_eager_execution(). Some of the
      * client side code was creating the initializer at the code load time, which
      * triggers the creation of RandomGenerator. Lazy init this class to walkaround
      * this issue until it is resolved on TF side.
      */
    private[keras] def maybeInit(tf: Ops): Unit = {
      // TODO(b/167482354): Change this back to normal init when the bug is fixed.
      if (built) return

      //if (tf.compat.v1.executing_eagerly_outside_functions() &&
      //  (use_generator_for_rng() || forceGenerator)) {
      //  // In the case of V2, we use tf.random.Generator to create all the random
      //  // numbers and seeds.
      //  import keras.utils.tf_utils // pylint: disable=g-import-not-at-top
      //  with tf_utils.maybe_init_scope(self):
      //  if (seed.isDefined) {
      //    self._generator = tf.random.Generator.from_seed(seed)
      //  } else {
      //    self._generator = tf.random.Generator.from_seed(random.randint(1, 1e9))
      //  }
      // } else {
        // In the v1 case, we use stateful op, regardless whether user provide a
        // seed or not. Seeded stateful op will ensure generating same sequences.
      // self._generator = None
      built = true
      // }
    }
  }
}
