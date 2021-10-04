// converted and adapted from TensorFlow; originally published under Apache 2.0 license
// Scala code published under LGPL v2.1+

package org.tensorflow.keras.layers

import org.tensorflow.Operand
import org.tensorflow.keras.activations.Activations
import org.tensorflow.keras.initializers.{Initializer, Initializers}
import org.tensorflow.keras.layers.Conv.Padding
import org.tensorflow.keras.utils.TensorShape
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.types.TFloat32

object Conv {
  object Padding extends Enumeration {
    val Valid, Same, Causal = Value
  }
  type Padding = Padding.Value

  object DataFormat extends Enumeration {
    val ChannelsLast, ChannelsFirst = Value
  }
  type DataFormat = DataFormat.Value

  trait Options {
    def groups              : Int
    def strides             : Seq[Int]
    def padding             : Padding
    def useBias             : Boolean
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
class Conv(rank: Int, filters: Int, kernelSize: Seq[Int], options: Conv.Options)
  extends Layer[TFloat32](1) {

  type T = TFloat32

  private val _isCausal = options.padding == Padding.Causal

  private var kernel: Variable[T] = _
  private var bias  : Variable[T] = _

  validateInit()

  private def validateInit(): Unit = {
    if (filters != 0 && filters % options.groups != 0)
      throw new IllegalArgumentException(
        s"The number of filters must be evenly divisible by the number of groups. Received: groups=${options.groups}, filters=${filters}"
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

  override protected def build(tf: Ops, inputShape: Shape): Unit = {
    val input_shape   = new TensorShape(inputShape)
    val input_channel: Int = ??? // this._get_input_channel(input_shape)
    if (input_channel % options.groups != 0)
      throw new IllegalArgumentException(
        "The number of input channels must be evenly divisible by the number " ++
        s"of groups. Received groups=${options.groups}, but the input has ${input_channel} channels " ++
        s"(full input shape is ${input_shape})."
      )

    ???
//    val kernel_shape = kernelSize + (input_channel / options.groups, this.filters)

    // compute_output_shape contains some validation logic for the input shape,
    // and make sure the output shape has all positive dimensions.
    ???
//    computeOutputShape(input_shape)

//    kernel = addWeight(tf,
//      name = "kernel",
//      shape = kernel_shape,
//      initializer = options.kernelInitializer,
//      regularizer = options.kernelRegularizer,
//      constraint = options.kernelConstraint,
//      trainable = true,
//      dtype = this.dtype
//    )
    ???
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

    val channel_axis: Int = ??? // this._get_channel_axis()
//    this.input_spec = new InputSpec(min_ndim=rank + 2, axes={channel_axis: input_channel})
    ???
    built = true
  }

  override def computeOutputShape(inputShape: Shape): Shape = ???

  override protected def call(tf: Ops, inputs: Operand[TFloat32]*): Operand[TFloat32] = ???
}

object Conv2D {
  val   Padding     = Conv.Padding
  type  Padding     = Padding.Value

  val   DataFormat  = Conv.DataFormat
  type  DataFormat  = DataFormat.Value

//  def setStrides(strideHeight: Long, strideWidth: Long): Options.Builder = {

  case class Options(
                      strides             : Seq[Int]            = Seq(1, 1),
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

class Conv2D(filters: Int, kernelSize: Seq[Int], options: Conv2D.Options)
  extends Conv(rank = 1, filters = filters, kernelSize = kernelSize, options = options) {

  private[layers] val convOp = null

//  override def build(tf: Ops, inputShape: Shape): Unit = ...

  override def computeOutputShape(inputShape: Shape) =
    throw new NotImplementedError("Not yet implemented") // XXX TODO

  @SafeVarargs override final protected def call(tf: Ops, inputs: Operand[TFloat32]*): Operand[TFloat32] =
    this.call(tf, inputs(0))

  private def call(tf: Ops, input: Operand[TFloat32]) =
    throw new NotImplementedError("Not yet implemented")
}