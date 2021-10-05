package org.tensorflow.keras.layers

import org.tensorflow.keras.activations.Activations
import org.tensorflow.keras.initializers.{Initializer, Initializers}
import org.tensorflow.keras.layers.Conv.{DataFormat, Padding}

object Conv2D {
  val   Padding   : Conv.Padding.type     = Conv.Padding
  type  Padding                           = Conv.Padding

  val   DataFormat: Conv.DataFormat.type  = Conv.DataFormat
  type  DataFormat                        = Conv.DataFormat
}

/** 2D convolution layer (e.g. spatial convolution over images).
  * This layer creates a convolution kernel that is convolved
  * with the layer input to produce a tensor of
  * outputs. If `use_bias` is True,
  * a bias vector is created and added to the outputs. Finally, if
  * `activation` is not `None`, it is applied to the outputs as well.
  * When using this layer as the first layer in a model,
  * provide the keyword argument `input_shape`
  * (tuple of integers or `None`, does not include the sample axis),
  * e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  * in `data_format="channels_last"`. You can use `None` when
  * a dimension has variable size.
  *
  * @param filters    the dimensionality of the output space (i.e. the number of
  *                   output filters in the convolution)
  * @param kernelSize two integers, specifying the height
  *                   and width of the 2D convolution window. Can be a single integer to specify
  *                   the same value for all spatial dimensions.
  */
class Conv2D(
              filters             : Int,
              kernelSize          : (Long, Long),
              strides             : (Long, Long)        = (1L, 1L),
              padding             : Padding             = Padding.Valid,
              dataFormat          : DataFormat          = DataFormat.ChannelsLast,
              dilationRate        : (Long, Long)        = (1L, 1L),
              groups              : Int                 = 1,
              activation          : Option[Activations] = None, // Activations.select(Activations.linear),
              useBias             : Boolean             = true,
              kernelInitializer   : Initializer         = Initializers.select(Initializers.glorotUniform),
              biasInitializer     : Initializer         = Initializers.select(Initializers.zeros),
              kernelRegularizer   : Option[Nothing]     = None,
              biasRegularizer     : Option[Nothing]     = None,
              activityRegularizer : Option[Nothing]     = None,
              kernelConstraint    : Option[Nothing]     = None,
              biasConstraint      : Option[Nothing]     = None,
            )
  extends Conv(rank = 1,
    filters           = filters,
    kernelSize        = Seq(kernelSize._1, kernelSize._2),
    strides           = Seq(strides._1, strides._2),
    padding           = padding,
    dataFormat        = dataFormat,
    dilationRate      = Seq(dilationRate._1, dilationRate._2),
    groups            = groups,
    activation        = activation,
    useBias           = useBias,
    kernelInitializer = kernelInitializer,
    biasInitializer   = biasInitializer,
    kernelRegularizer = kernelRegularizer,
    biasRegularizer   = biasRegularizer,
    activityRegularizer = activityRegularizer,
    kernelConstraint = kernelConstraint,
    biasConstraint    = biasConstraint,
  )
