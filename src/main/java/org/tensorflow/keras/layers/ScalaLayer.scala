package org.tensorflow.keras.layers

import org.tensorflow.op.core.Variable
import org.tensorflow.types.family.{TNumber, TType}

trait ScalaLayer[T <: TNumber] {
  this: Layer[T] =>

  /** Adds a new variable to the layer.

     *  trainable: Boolean,
     *  constraint:
     *  use_resource: Whether to use `ResourceVariable`.
     *  synchronization: Indicates when a distributed a variable will be
     *    aggregated. Accepted values are constants defined in the class
     *    `tf.VariableSynchronization`. By default the synchronization is set to
     *    `AUTO` and the current `DistributionStrategy` chooses
     *    when to synchronize. If `synchronization` is set to `ON_READ`,
     *    `trainable` must not be set to `True`.
     *  aggregation: Indicates how a distributed variable will be aggregated.
     *    Accepted values are constants defined in the class
     *    `tf.VariableAggregation`.
     ***kwargs: Additional keyword arguments. Accepted values are `getter`,
     *    `collections`, `experimental_autocast` and `caching_device`.
    * @param name             Variable name.
    * @param shape            Variable shape. Defaults to scalar if unspecified.
    * @param dtype            The type of the variable. Defaults to `self.dtype`.
    * @param initializer      Initializer instance (function).
    * @param regularizer      Regularizer instance (function).
    * @param trainable        whether the variable should be part of the layer's
    *    "trainable_variables" (e.g. variables, biases)
    *    or "non_trainable_variables" (e.g. BatchNorm mean and variance).
    *    Note that `trainable` cannot be `True` if `synchronization`
    *    is set to `ON_READ`.
    * @param constraint       Constraint instance (function).
    * @param use_resource
    * @param synchronization
    * @param aggregation
    * @return
    */
  protected final def addWeightExt(
                                    name    : Option[String] = None,
                                    shape=None,
                                    dtype=None,
                                    initializer=None,
                                    regularizer=None,
                                    trainable : Option[Boolean] = None,
                                    constraint=None,
                                    use_resource=None,
                                    synchronization=tf.VariableSynchronization.AUTO,
                                    aggregation=tf.VariableAggregation.NONE,
                                  ): Variable[T] = {
    ???
  }
}
