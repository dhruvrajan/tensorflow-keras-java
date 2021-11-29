package org.tensorflow.keras.layers

import org.tensorflow.framework.initializers.Initializer
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.core.Variable
import org.tensorflow.proto.framework.{VariableAggregation, VariableSynchronization}
import org.tensorflow.types.family.TNumber

trait ScalaLayer[T <: TNumber] {
  self: Layer[T] =>

  /** Adds a new variable to the layer.
    *
    * kwargs: Additional keyword arguments. Accepted values are `getter`,
    *    `collections`, `experimental_autocast` and `caching_device`.
    *
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
    * @param use_resource     Whether to use `ResourceVariable`.
    * @param synchronization  Indicates when a distributed a variable will be
    *    aggregated. Accepted values are constants defined in the class
    *    `tf.VariableSynchronization`. By default the synchronization is set to
    *    `AUTO` and the current `DistributionStrategy` chooses
    *    when to synchronize. If `synchronization` is set to `ON_READ`,
    *    `trainable` must not be set to `True`.
    * @param aggregation      Indicates how a distributed variable will be aggregated.
    *    Accepted values are constants defined in the class
    *    `tf.VariableAggregation`.
    * @return
    */
  protected final def addWeightExt(
                                    name            : String, //                 = None,
                                    shape           : Shape                   = Shape.scalar() /*unknown()*/,
                                    dtype           : Class[_ <: TNumber]     = self.dtype,
                                    initializer     : Option[Initializer[T]]  = None,
                                    regularizer     : Option[Nothing]         = None,
                                    trainable       : Option[Boolean]         = None,
                                    constraint      : Option[Nothing]         = None,
                                    use_resource    : Boolean                 = false,
                                    synchronization: VariableSynchronization  = VariableSynchronization.VARIABLE_SYNCHRONIZATION_AUTO,
                                    aggregation     : VariableAggregation     = VariableAggregation.VARIABLE_AGGREGATION_NONE,
                                  ): Variable[T] = {


    ???
  }
}
