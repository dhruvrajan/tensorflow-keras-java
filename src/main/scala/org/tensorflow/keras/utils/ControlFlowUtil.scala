package org.tensorflow.keras.utils

import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.types.family.TType

object ControlFlowUtil {
  /** Returns either `true_fn()` if predicate `pred` is true else `false_fn()`.
    * If `pred` is a bool or has a constant value, we return either `true_fn()`
    * or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.
    * Args:
    * @param   pred A scalar determining whether to return the result of `true_fn` or
    *     `false_fn`.
    * @param true_fn The function to be performed if pred is true.
    * @param false_fn: The function to be performed if pred is false.
    *   name: Optional name prefix when using `tf.cond`.
    * @return
    *   Tensors returned by the call to either `true_fn` or `false_fn`.
    * @throws IllegalArgumentException If `true_fn` or `false_fn` is not callable.
    */
  def smartCond[T <: TType](tf: Ops, pred: Variable[T], true_fn: Any = None, false_fn: Any = None /*, name=None*/) = { // pylint: disable=invalid-name
    // XXX TODO where is in tf-java?
//     tf.cond(
//      pred, true_fn=true_fn, false_fn=false_fn, name=name)
//     return tf.__internal__.smart_cond.smart_cond(
//       pred, true_fn=true_fn, false_fn=false_fn, name=name)
    ???
  }
}
