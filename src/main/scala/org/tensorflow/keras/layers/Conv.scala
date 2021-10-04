// converted and adapted from TensorFlow; originally published under Apache 2.0 license
// Scala code published under LGPL v2.1+

package org.tensorflow.keras.layers

import org.tensorflow.keras.activations.Activations
import org.tensorflow.keras.initializers.{Initializer, Initializers}
import org.tensorflow.keras.layers.Conv.{DataFormat, Padding}
import org.tensorflow.keras.utilss.ConvUtils
import org.tensorflow.ndarray.Shape
import org.tensorflow.{Operand, op}
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.op.nn.BiasAdd
import org.tensorflow.types.TFloat32

import scala.collection.JavaConverters.seqAsJavaListConverter

object Conv {
  object Padding {
    case object Valid  extends Padding { val tfName = "VALID"  }
    case object Same   extends Padding { val tfName = "SAME"   }
    case object Causal extends Padding { val tfName = "CAUSAL" }
    case object Full   extends Padding { val tfName = "FULL"   }
  }
  sealed trait Padding { def tfName: String }

  object DataFormat {
    case object ChannelsLast  extends DataFormat {
      override def tfName(ndim: Int): String = ndim match {
        case 3 => "NWC"
        case 4 => "NHWC"
        case 5 => "NDHWC"
        case _ =>
          throw new IllegalArgumentException(
            s"Input rank not supported: $ndim. Expected values are [3, 4, 5]"
          )
      }
    }
    case object ChannelsFirst extends DataFormat {
      override def tfName(ndim: Int): String = ndim match {
        case 3 => "NCW"
        case 4 => "NCHW"
        case 5 => "NCDHW"
        case _ =>
          throw new IllegalArgumentException(
            s"Input rank not supported: $ndim. Expected values are [3, 4, 5]"
          )
      }
    }
  }
  sealed trait DataFormat { def tfName(ndim: Int): String }

  trait Options {
    def groups              : Int
    def strides             : Seq[Long]
    def padding             : Padding
    def useBias             : Boolean
    def dataFormat          : DataFormat
    def dilationRate        : Seq[Long]
  }
}
/** Abstract N-D convolution layer (private, used as implementation base).
  * This layer creates a convolution kernel that is convolved
  * (actually cross-correlated) with the layer input to produce a tensor of
  * outputs. If `useBias` is true (and a `biasInitializer` is provided),
  * a bias vector is created and added to the outputs. Finally, if
  * `activation` is not `None`, it is applied to the outputs as well.
  * Note: layer attributes cannot be modified after the layer has been called
  * once (except the `trainable` attribute)
  *
  * @param  rank  the rank of the convolution, e.g. `2` for 2D convolution.
  */
class Conv(rank: Int, filters: Int, kernelSize: Seq[Long], options: Conv.Options)
  extends Layer[TFloat32](1) {

  type T = TFloat32

  private val isCausal        = options.padding == Padding.Causal
  private val tfDataFormat    = options.dataFormat.tfName(rank + 2)
  private val isChannelsFirst = options.dataFormat == DataFormat.ChannelsFirst

  private var kernel: Variable[T] = _
  private var bias  : Variable[T] = _

  validateInit()

  private def validateInit(): Unit = {
    if (filters != 0 && filters % options.groups != 0)
      throw new IllegalArgumentException(
        "The number of filters must be evenly divisible by the number of groups. " ++
        s"Received: groups=${options.groups}, filters=$filters"
      )

    if (!kernelSize.forall(_ > 0))
      throw new IllegalArgumentException(
        s"The argument `kernelSize` cannot contain 0(s). Received: $kernelSize"
      )

    if (!options.strides.forall(_ > 0))
      throw new IllegalArgumentException(
        s"The argument `strides` cannot contains 0(s). Received: ${options.strides}"
      )

    if (options.padding == Padding.Causal && !(rank == 1))
      throw new IllegalArgumentException(
        "Causal padding is only supported for `Conv1D` and `SeparableConv1D`."
      )
  }

  private def getChannelAxis: Int =
    if (options.dataFormat == DataFormat.ChannelsFirst) -1 - rank else -1

  private def getInputChannel(inputShape: Shape): Int = {
    val channelAxis = getChannelAxis
    val value = inputShape.size(channelAxis)
    if (value == Shape.UNKNOWN_SIZE)
      throw new IllegalArgumentException(
        "The channel dimension of the inputs should be defined. " ++
        s"The input_shape received is $inputShape, " ++
        s"where axis $channelAxis (0-based) " ++
        "is the channel dimension, which found to be `None`.")
    value.toInt
  }

  override protected def build(tf: Ops, inputShape: Shape): Unit = {
    val inputChannel = getInputChannel(inputShape)
    if (inputChannel % options.groups != 0)
      throw new IllegalArgumentException(
        "The number of input channels must be evenly divisible by the number " ++
        s"of groups. Received groups=${options.groups}, but the input has $inputChannel channels " ++
        s"(full input shape is $inputShape)."
      )

    require (kernelSize.size == rank)
    val kernelShape: Array[Long] = kernelSize.toArray[Long] ++ Array[Long](inputChannel / options.groups, this.filters)

    // compute_output_shape contains some validation logic for the input shape,
    // and make sure the output shape has all positive dimensions.
    computeOutputShape(inputShape)

    // XXX TODO
    kernel = addWeight(
      "kernel",
      tf.variable(Shape.of(kernelShape: _*), dtype): Variable[T], // XXX TODO
//      /*initializer =*/ options.kernelInitializer,
//      /*regularizer =*/ options.kernelRegularizer,
//      /*constraint  =*/ options.kernelConstraint,
//      trainable   = true,
//      dtype       = this.dtype
    )
    if (options.useBias) {
      ???
//      bias = addWeight(tf,
//        name = "bias",
//        shape = (filters,),
//        initializer = options.biasInitializer,
//        regularizer = options.biasRegularizer,
//        constraint = options.biasConstraint,
//        trainable = true,
//        dtype = this.dtype
//      )
    }
    //    else {
    //      self.bias = None
    //    }

    val channelAxis = getChannelAxis
    //    this.input_spec = new InputSpec(min_ndim=rank + 2, axes={channel_axis: input_channel})
    ???

    built = true
  }

  private def spatialOutputShape(spatialInputShape: Shape): Shape = {
    val arr = Array.tabulate(spatialInputShape.numDimensions()) { i =>
      val length = spatialInputShape.size(i)
      ConvUtils.convOutputLength(
        length,
        kernelSize(i),
        padding   = options.padding,
        stride    = options.strides(i),
        dilation  = options.dilationRate(i)
      )
    }
    Shape.of(arr: _*)
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
//    val inputShape = tf.TensorShape(inputShape).as_list()
    val batchRank = inputShape.numDimensions() - rank - 1
    try
      if (options.dataFormat == DataFormat.ChannelsLast)
        // Shape.of(
          inputShape.take(batchRank) append
          spatialOutputShape(inputShape.subShape(batchRank, inputShape.numDimensions() - 1)) append
          filters
        // )
      else
        // Shape.of(
          inputShape.take(batchRank) append filters append
          spatialOutputShape(inputShape.subShape(batchRank + 1, inputShape.numDimensions()))
        // )

    catch {
      case ex: IllegalArgumentException =>
        throw new IllegalArgumentException(
          "One of the dimensions in the output is <= 0 " ++
          "due to downsampling. Consider " ++
          "increasing the input size. " ++
          s"Received input shape $inputShape which would produce " ++
          "output shape with a zero or negative value in a dimension.", ex
        )
    }
  }

  def convolutionOp(tf: Ops, inputs: Operand[T], kernel: Operand[T]): Operand[T] = {
    val tfPadding =
      if (options.padding == Padding.Causal)
        Padding.Valid.tfName   // Causal padding handled in `call`.
      else options.padding.tfName

    tf.nn.conv2d /*convolution*/(
      inputs,
      kernel,
      options.strides.map(_.asInstanceOf[java.lang.Long]).asJava,
      tfPadding,
      op.nn.Conv2d
        .dilations(options.dilationRate.map(_.asInstanceOf[java.lang.Long]).asJava)
        .dataFormat(tfDataFormat)
    )
  }

  @SafeVarargs override final protected def call(tf: Ops, inputs: Operand[T]*): Operand[T] =
    call(tf, inputs(0))

  // Calculates padding for 'causal' option for 1-d conv layers.
  private def computeCausalPadding(inputs: Operand[T]) = {
    val leftPad   = options.dilationRate.head * (kernelSize.head - 1)
    val batchRank = if (inputs.shape.numDimensions() == 0 /*None*/)
      1
    else
      inputs.shape.numDimensions() - 2

    ???
//    val causal_padding = if (options.dataFormat == DataFormat.ChannelsLast)
//      [[0, 0]] * batchRank + [[leftPad, 0], [0, 0]]
//    else
//      [[0, 0]] * batchRank + [[0, 0], [leftPad, 0]]
//     causal_padding
  }

  private def call(tf: Ops, inputs0: Operand[T]): Operand[T] = {
    var inputs      = inputs0
    val input_shape = inputs.shape

    if (isCausal) // Apply causal padding to inputs for Conv1D.
      inputs = tf.pad(inputs, computeCausalPadding(inputs), tf.constant(0f))

    var outputs: Operand[T] = convolutionOp(tf, inputs, kernel)

    if (options.useBias) {
      val outputRank = outputs.shape.numDimensions()

      if (rank == 1 && isChannelsFirst) {
        // nn.bias_add does not accept a 1D input tensor.
        val bias = tf.reshape(this.bias, tf.constant(Array(1, filters, 1)))

        // outputs += bias
        outputs = tf.math.add(outputs, bias)  // XXX TODO correct?
      } else {
        def applyFn(o: Operand[T]): Operand[T] =
          tf.nn.biasAdd(o, this.bias, BiasAdd.dataFormat(tfDataFormat))

        // Handle multiple batch dimensions.
        if (outputRank > 0 /*None*/ && outputRank > 2 + rank) {
          outputs = ConvUtils.squeezeBatchDims(tf,
            outputs, applyFn, innerRank = rank + 1)
        } else {
          outputs = applyFn(outputs)
        }
      }
    }
    if (!tf.scope().env().isEager) {
      // Infer the static output shape:
      val outShape = computeOutputShape(input_shape)
      ??? // outputs.set_shape(outShape)
    }

// XXX TODO: activation
//    if (options.activation != None)
//      this.activation(outputs)
//    else
      outputs
  }
}

object Conv2D {
  val   Padding     = Conv.Padding
  type  Padding     = Conv.Padding // Padding.Value

  val   DataFormat  = Conv.DataFormat
  type  DataFormat  = Conv.DataFormat

//  def setStrides(strideHeight: Long, strideWidth: Long): Options.Builder = {

  case class Options(
                      strides             : Seq[Long]           = Seq(1, 1),
                      padding             : Padding             = Padding.Valid,
                      dataFormat          : DataFormat          = DataFormat.ChannelsLast,
                      dilationRate        : Seq[Long]           = Seq[Long](1, 1),
                      groups              : Int                 = 1,
                      activation          : Option[Activations] = None, // Activations.select(Activations.linear),
                      useBias             : Boolean             = true,
                      kernelInitializer   : Initializer         = Initializers.select(Initializers.glorotUniform),
                      biasInitializer     : Initializer         = Initializers.select(Initializers.zeros),
//                      kernelRegularizer   : Regularizer         = ...,
//                      biasRegularizer     : Regularizer         = ...,
//                      activityRegularizer : Regularizer         = ...,
//                      kernelConstraint    : Constraint          = ...,
//                      biasConstraint      : Constraint          = ...,
                    ) extends Conv.Options
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
class Conv2D(filters: Int, kernelSize: Seq[Long], options: Conv2D.Options)
  extends Conv(rank = 1, filters = filters, kernelSize = kernelSize, options = options)
