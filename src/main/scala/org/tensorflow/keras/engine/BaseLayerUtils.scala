package org.tensorflow.keras.engine

import org.tensorflow.proto.framework.{VariableAggregation, VariableSynchronization}
import org.tensorflow.types.TFloat32
import org.tensorflow.types.family.TNumber

object BaseLayerUtils {
  def makeVariable(name: String,
    shape  : Option[Nothing]  =None,
    dtype : Class[_ <: TNumber] = classOf[TFloat32],
    initializer : Option[Nothing] = None,
    trainable   : Option[Boolean] = None,
    caching_device : Option[Nothing]  =None,
    validate_shape: Boolean =true,
    constraint  : Option[Nothing]  =None,
    use_resource  : Option[Nothing]  =None,
    collections  : Option[Nothing] =None,
    synchronization : VariableSynchronization = VariableSynchronization .VARIABLE_SYNCHRONIZATION_AUTO,
    aggregation     : VariableAggregation     = VariableAggregation     .VARIABLE_AGGREGATION_NONE,
    partitioner : Option[Nothing] =None) = {

    ???
  }
}
