package org.tensorflow.keras.layers

import org.tensorflow.Tensor
import org.tensorflow.framework.initializers.Initializer
import org.tensorflow.keras.activations.Activations
import org.tensorflow.keras.initializers.Initializers
import org.tensorflow.keras.layers.BatchNormalization.RenormClipping
import org.tensorflow.keras.layers.Conv.{DataFormat, Padding}
import org.tensorflow.keras.utils.Keras
import org.tensorflow.ndarray.Shape
import org.tensorflow.types.TFloat32
import org.tensorflow.types.family.TNumber

object Layers {
  def input[T <: TNumber](units: Shape, batchSize: Long = Shape.UNKNOWN_SIZE) =
    new Input[T](units /*Keras.concatenate(firstDim, units: _*)*/, batchSize = batchSize)

  def dense[T <: TNumber](units: Int,
                          activation          : Option[Activations] = None,
                          useBias             : Boolean = true,
                          kernelInitializer   : Initializers.Value = Initializers.glorotUniform,
                          biasInitializer     : Initializers.Value = Initializers.zeros,
                          kernelRegularizer   : Option[Nothing] = None,
                          biasRegularizer     : Option[Nothing] = None,
                          activityRegularizer : Option[Nothing] = None,
                          kernelConstraint    : Option[Nothing] = None,
                          biasConstraint      : Option[Nothing] = None,
                         ): Dense[T] =
    new Dense[T](
      units             = units,
      activation        = activation.map(Activations.select[T]),
      useBias           = useBias,
      kernelInitializer = Initializers.select[T](kernelInitializer),
      biasInitializer   = Initializers.select[T](biasInitializer),
      kernelRegularizer = kernelRegularizer,
      biasRegularizer   = biasRegularizer,
      activityRegularizer = activityRegularizer,
      kernelConstraint = kernelConstraint,
      biasConstraint = biasConstraint
    )

  // Builders for Flatten Layer
  def flatten[T <: TNumber]() = new Flatten[T]

  // Builders for Conv2D Layer
  def conv2D(
              filters             : Int,
              kernelSize          : (Long, Long),
              strides             : (Long, Long)        = (1L, 1L),
              padding             : Padding             = Padding.Valid,
              dataFormat          : DataFormat          = DataFormat.ChannelsLast,
              dilationRate        : (Long, Long)        = (1L, 1L),
              groups              : Int                 = 1,
              activation          : Option[Activations] = None, // Activations.select(Activations.linear),
              useBias             : Boolean             = true,
              kernelInitializer   : Initializer[TFloat32] = Initializers.select[TFloat32](Initializers.glorotUniform),
              biasInitializer     : Initializer[TFloat32] = Initializers.select[TFloat32](Initializers.zeros),
              kernelRegularizer   : Option[Nothing]     = None,
              biasRegularizer     : Option[Nothing]     = None,
              activityRegularizer : Option[Nothing]     = None,
              kernelConstraint    : Option[Nothing]     = None,
              biasConstraint      : Option[Nothing]     = None,
            ) =
    new Conv2D(
      filters             = filters,
      kernelSize          = kernelSize,
      strides             = strides,
      padding             = padding,
      dataFormat          = dataFormat,
      dilationRate        = dilationRate,
      groups              = groups,
      activation          = activation,
      useBias             = useBias,
      kernelInitializer   = kernelInitializer,
      biasInitializer     = biasInitializer,
      kernelRegularizer   = kernelRegularizer,
      biasRegularizer     = biasRegularizer,
      activityRegularizer = activityRegularizer,
      kernelConstraint    = kernelConstraint,
      biasConstraint      = biasConstraint,
    )

  def conv2DTranspose(
                       filters            : Int,
                       kernelSize         : (Long, Long),
                       strides            : (Long, Long)          = (1L, 1L),
                       padding            : Conv.Padding          = Conv.Padding.Valid,
                       outputPadding      : Option[(Long, Long)]  = None,
                       data_format        : DataFormat            = DataFormat.ChannelsLast,
                       dilation_rate      : (Long, Long)          = (1L, 1L),
                       activation         : Option[Activations]   = None,
                       useBias            : Boolean               = true,
                       kernelInitializer  : Initializer[TFloat32] = Initializers.select(Initializers.glorotUniform),
                       biasInitializer    : Initializer[TFloat32] = Initializers.select(Initializers.zeros),
                       kernelRegularizer  : Option[Nothing]       = None,
                       biasRegularizer    : Option[Nothing]       = None,
                       activityRegularizer: Option[Nothing]       = None,
                       kernelConstraint   : Option[Nothing]       = None,
                       biasConstraint     : Option[Nothing]       = None,
  ) =
    new Conv2DTranspose(
      filters               = filters            ,
      kernelSize           = kernelSize         ,
      strides               = strides            ,
      padding               = padding            ,
      outputPadding        = outputPadding      ,
      dataFormat           = data_format        ,
      dilationRate         = dilation_rate      ,
      activation            = activation         ,
      useBias              = useBias            ,
      kernelInitializer    = kernelInitializer  ,
      biasInitializer      = biasInitializer    ,
      kernelRegularizer    = kernelRegularizer  ,
      biasRegularizer      = biasRegularizer    ,
      activityRegularizer  = activityRegularizer,
      kernelConstraint     = kernelConstraint   ,
      biasConstraint       = biasConstraint     ,
    )

  // Builders for LeakyReLU Layer
  def leakyReLU[T <: TNumber](alpha: Float = 0.3f) = new LeakyReLU[T](alpha = alpha)

  // Builders for Dropout Layer
  def dropout[T <: TNumber](rate: Float, noiseShape: Shape = Shape.unknown(), seed: Option[Long] = None) =
    new Dropout[T](rate = rate, noiseShape = noiseShape, seed = seed)

  // Builders for BatchNormalization Layer
  def batchNormalization[T <: TNumber](
                          axis0           : Seq[Int]        = Seq(-1),
                          momentum        : Float           = 0.99f,
                          epsilon         : Float           = 1e-3f,
                          center          : Boolean         = true,
                          scale           : Boolean         = true,
                          betaInitializer : Initializer[T] = Initializers.select[T](Initializers.zeros),
                          gammaInitializer: Initializer[T] = Initializers.select[T](Initializers.ones ),
                          movingMeanInitializer: Initializer[T] = Initializers.select[T](Initializers.zeros),
                          movingVarianceInitializer: Initializer[T] = Initializers.select[T](Initializers.ones ),
                          betaRegularizer : Option[Nothing] = None,
                          gammaRegularizer: Option[Nothing] = None,
                          betaConstraint  : Option[Nothing] = None,
                          gammaConstraint : Option[Nothing] = None,
                          renorm          : Boolean         = false,
                          renormClipping  : Map[RenormClipping, Tensor] = Map.empty,
                          renormMomentum  : Float           = 0.99f,
                          fused0          : Option[Boolean] = None,
                          trainable       : Boolean         = true,
                          virtualBatchSize: Option[Int]     = None,
                          adjustment      : Option[Nothing] = None,
                          //  name=None,
          ) =
    new BatchNormalization[T](
      axis0                     = axis0                     ,
      momentum                  = momentum                  ,
      epsilon                   = epsilon                   ,
      center                    = center                    ,
      scale                     = scale                     ,
      betaInitializer           = betaInitializer           ,
      gammaInitializer          = gammaInitializer          ,
      movingMeanInitializer     = movingMeanInitializer     ,
      movingVarianceInitializer = movingVarianceInitializer ,
      betaRegularizer           = betaRegularizer           ,
      gammaRegularizer          = gammaRegularizer          ,
      betaConstraint            = betaConstraint            ,
      gammaConstraint           = gammaConstraint           ,
      renorm                    = renorm                    ,
      renormClipping            = renormClipping            ,
      renormMomentum            = renormMomentum            ,
      fused0                    = fused0                    ,
      trainable                 = trainable                 ,
      virtualBatchSize          = virtualBatchSize          ,
      adjustment                = adjustment                ,
    )

  // Builders for Reshape Layer
  def reshape[T <: TNumber](targetShape: Shape) = new Reshape[T](targetShape)
}
