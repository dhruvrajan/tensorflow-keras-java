// converted and adapted from TensorFlow; originally published under Apache 2.0 license
// Scala code published under LGPL v2.1+

package org.tensorflow.keras.layers

import org.tensorflow.keras.activations.Activations
import org.tensorflow.keras.initializers.Initializer
import org.tensorflow.keras.layers.Conv.{DataFormat, Padding}
import org.tensorflow.keras.utils.ConvUtils
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.op.nn.BiasAdd
import org.tensorflow.types.TFloat32
import org.tensorflow.{Operand, op}

import scala.collection.JavaConverters._

object Conv {
  object Padding {
    case object Valid  extends Padding { val tfName = "VALID"  }
    case object Same   extends Padding { val tfName = "SAME"   }
    case object Causal extends Padding { val tfName = "CAUSAL" }
    case object Full   extends Padding { val tfName = "FULL"   }
  }
  sealed trait Padding { def tfName: String }

  object DataFormat {
    case object ChannelsLast extends DataFormat {
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
class Conv(
            rank                : Int,
            filters             : Int,
            kernelSize          : Seq[Long],
            strides             : Seq[Long],
            padding             : Padding,
            dataFormat          : DataFormat,
            dilationRate        : Seq[Long],
            groups              : Int,
            activation          : Option[Activations],
            useBias             : Boolean,
            kernelInitializer   : Initializer,
            biasInitializer     : Initializer,
            kernelRegularizer   : Option[Nothing],
            biasRegularizer     : Option[Nothing],
            activityRegularizer : Option[Nothing],
            kernelConstraint    : Option[Nothing],
            biasConstraint      : Option[Nothing],
            trainable           : Boolean = true,
//            convOp              : Option[Nothing],
          )
  extends Layer[TFloat32](1) with ScalaLayer[TFloat32] {

  type T = TFloat32

  private val isCausal        = /*options.*/padding == Padding.Causal
  private val tfDataFormat    = /*options.*/dataFormat.tfName(rank + 2)
  private val isChannelsFirst = /*options.*/dataFormat == DataFormat.ChannelsFirst

  private var kernel: Variable[T] = _
  private var bias  : Variable[T] = _

  validateInit()

  private def validateInit(): Unit = {
    if (filters != 0 && filters % groups != 0)
      throw new IllegalArgumentException(
        "The number of filters must be evenly divisible by the number of groups. " ++
        s"Received: groups=$groups, filters=$filters"
      )

    if (!kernelSize.forall(_ > 0))
      throw new IllegalArgumentException(
        s"The argument `kernelSize` cannot contain 0(s). Received: $kernelSize"
      )

    if (!strides.forall(_ > 0))
      throw new IllegalArgumentException(
        s"The argument `strides` cannot contains 0(s). Received: $strides"
      )

    if (padding == Padding.Causal && !(rank == 1))
      throw new IllegalArgumentException(
        "Causal padding is only supported for `Conv1D` and `SeparableConv1D`."
      )
  }

  private def getChannelAxis: Int =
    if (dataFormat == DataFormat.ChannelsFirst) -1 - rank else -1

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
    if (inputChannel % groups != 0)
      throw new IllegalArgumentException(
        "The number of input channels must be evenly divisible by the number " ++
        s"of groups. Received groups=$groups, but the input has $inputChannel channels " ++
        s"(full input shape is $inputShape)."
      )

    require (kernelSize.size == rank)
    val kernelShape: Array[Long] = kernelSize.toArray[Long] ++ Array[Long](inputChannel / groups, this.filters)

    // compute_output_shape contains some validation logic for the input shape,
    // and make sure the output shape has all positive dimensions.
    computeOutputShape(inputShape)

    kernel = addWeightExt(
      name        = "kernel",
      shape       = Shape.of(kernelShape: _*),
      initializer = Some(kernelInitializer),
      regularizer = kernelRegularizer,
      constraint  = kernelConstraint,
      trainable   = Some(true),
      dtype       = this.dtype
    )
    if (useBias) {
      ???
      // XXX TODO
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
        padding   = padding,
        stride    = strides(i),
        dilation  = dilationRate(i)
      )
    }
    Shape.of(arr: _*)
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
//    val inputShape = tf.TensorShape(inputShape).as_list()
    val batchRank = inputShape.numDimensions() - rank - 1
    try
      if (dataFormat == DataFormat.ChannelsLast)
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
      if (padding == Padding.Causal)
        Padding.Valid.tfName   // Causal padding handled in `call`.
      else padding.tfName

    tf.nn.conv2d /*convolution*/(
      inputs,
      kernel,
      strides.map(_.asInstanceOf[java.lang.Long]).asJava,
      tfPadding,
      op.nn.Conv2d
        .dilations(dilationRate.map(_.asInstanceOf[java.lang.Long]).asJava)
        .dataFormat(tfDataFormat)
    )
  }

  @SafeVarargs override final protected def call(tf: Ops, inputs: Operand[T]*): Operand[T] =
    callOne(tf, inputs(0))

  // Calculates padding for 'causal' option for 1-d conv layers.
  private def computeCausalPadding(inputs: Operand[T]) = {
    val leftPad   = dilationRate.head * (kernelSize.head - 1)
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

  private def callOne(tf: Ops, inputs0: Operand[T]): Operand[T] = {
    var inputs      = inputs0
    val input_shape = inputs.shape

    if (isCausal) // Apply causal padding to inputs for Conv1D.
      inputs = tf.pad(inputs, computeCausalPadding(inputs), tf.constant(0f))

    var outputs: Operand[T] = convolutionOp(tf, inputs, kernel)

    if (useBias) {
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

    if (activation.isDefined) {
      val a = Activations.select[T](activation.get)
      a.apply(tf, outputs)
    } else {
      outputs
    }
  }
}
