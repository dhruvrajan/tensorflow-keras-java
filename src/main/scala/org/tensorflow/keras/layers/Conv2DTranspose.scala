package org.tensorflow.keras.layers

import org.tensorflow.Operand
import org.tensorflow.framework.initializers.Initializer
import org.tensorflow.keras.activations.Activations
import org.tensorflow.keras.initializers.Initializers
import org.tensorflow.keras.layers.Conv.DataFormat
import org.tensorflow.keras.utils.ConvUtils
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32

// based on https://github.com/keras-team/keras/blob/master/keras/layers/convolutional/conv2d_transpose.py

/** Transposed convolution layer (sometimes called Deconvolution).
  * The need for transposed convolutions generally arises
  * from the desire to use a transformation going in the opposite direction
  * of a normal convolution, i.e., from something that has the shape of the
  * output of some convolution to something that has the shape of its input
  * while maintaining a connectivity pattern that is compatible with
  * said convolution.
  * When using this layer as the first layer in a model,
  * provide the keyword argument `input_shape`
  * (tuple of integers or `None`, does not include the sample axis),
  * e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  * in `data_format="channels_last"`.
  *
  * Input shape:
  *   4D tensor with shape:
  *   `(batch_size, channels, rows, cols)` if data_format='channels_first'
  *   or 4D tensor with shape:
  *   `(batch_size, rows, cols, channels)` if data_format='channels_last'.
  * Output shape:
  *   4D tensor with shape:
  *   `(batch_size, filters, new_rows, new_cols)` if data_format='channels_first'
  *   or 4D tensor with shape:
  *   `(batch_size, new_rows, new_cols, filters)` if data_format='channels_last'.
  *   `rows` and `cols` values might have changed due to padding.
  *   If `output_padding` is specified:
  *   {{{
  *   new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] +
  *   output_padding[0])
  *   new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] +
  *   output_padding[1])
  *   }}}
  *
  * References:
  *   - [A guide to convolution arithmetic for deep
  *     learning](https://arxiv.org/abs/1603.07285v1)
  *   - [Deconvolutional
  *     Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  *
  * @param filters  the dimensionality of the output space
  *     (i.e. the number of output filters in the convolution).
  * @param kernelSize  An integer or tuple/list of 2 integers, specifying the
  *     height and width of the 2D convolution window.
  *     Can be a single integer to specify the same value for
  *     all spatial dimensions.
  * @param strides  An integer or tuple/list of 2 integers,
  *     specifying the strides of the convolution along the height and width.
  *     Can be a single integer to specify the same value for
  *     all spatial dimensions.
  *     Specifying any stride value != 1 is incompatible with specifying
  *     any `dilation_rate` value != 1.
  * @param padding  one of `"valid"` or `"same"` (case-insensitive).
  *     `"valid"` means no padding. `"same"` results in padding with zeros evenly
  *     to the left/right or up/down of the input such that output has the same
  *     height/width dimension as the input.
  * @param outputPadding An integer or tuple/list of 2 integers,
  *     specifying the amount of padding along the height and width
  *     of the output tensor.
  *     Can be a single integer to specify the same value for all
  *     spatial dimensions.
  *     The amount of output padding along a given dimension must be
  *     lower than the stride along that same dimension.
  *     If set to `None` (default), the output shape is inferred.
  * @param dataFormat  A string,
  *     one of `channels_last` (default) or `channels_first`.
  *     The ordering of the dimensions in the inputs.
  *     `channels_last` corresponds to inputs with shape
  *     `(batch_size, height, width, channels)` while `channels_first`
  *     corresponds to inputs with shape
  *     `(batch_size, channels, height, width)`.
  *     It defaults to the `image_data_format` value found in your
  *     Keras config file at `~/.keras/keras.json`.
  *     If you never set it, then it will be "channels_last".
  * @param dilationRate  an integer or tuple/list of 2 integers, specifying
  *     the dilation rate to use for dilated convolution.
  *     Can be a single integer to specify the same value for
  *     all spatial dimensions.
  *     Currently, specifying any `dilation_rate` value != 1 is
  *     incompatible with specifying any stride value != 1.
  * @param activation Activation function to use.
  *     If you don't specify anything, no activation is applied
  *     (see `keras.activations`).
  * @param useBias Boolean, whether the layer uses a bias vector.
  * @param kernelInitializer Initializer for the `kernel` weights matrix
  *     (see `keras.initializers`). Defaults to 'glorot_uniform'.
  * @param biasInitializer Initializer for the bias vector
  *     (see `keras.initializers`). Defaults to 'zeros'.
  * @param kernelRegularizer Regularizer function applied to
  *     the `kernel` weights matrix (see `keras.regularizers`).
  * @param biasRegularizer Regularizer function applied to the bias vector
  *     (see `keras.regularizers`).
  * @param activityRegularizer Regularizer function applied to
  *     the output of the layer (its "activation") (see `keras.regularizers`).
  * @param kernelConstraint  Constraint function applied to the kernel matrix
  *     (see `keras.constraints`).
  * @param biasConstraint  Constraint function applied to the bias vector
  *     (see `keras.constraints`).
  *
  * @return  A tensor of rank 4 representing
  *   `activation(conv2dtranspose(inputs, kernel) + bias)`.
  *
  * @throws IllegalArgumentException  if `padding` is "causal".
  * @throws IllegalArgumentException  when both `strides` > 1 and `dilation_rate` > 1.
  */
class Conv2DTranspose(
                       filters            : Int,
                       kernelSize         : (Long, Long),
                       strides            : (Long, Long)          = (1L, 1L),
                       padding            : Conv.Padding          = Conv.Padding.Valid,
                       outputPadding      : Option[(Long, Long)]  = None,
                       dataFormat         : DataFormat            = DataFormat.ChannelsLast,
                       dilationRate       : (Long, Long)          = (1L, 1L),
                       activation         : Option[Activations]   = None,
                       useBias            : Boolean               = true,
                       kernelInitializer  : Initializer[TFloat32] = Initializers.select(Initializers.glorotUniform),
                       biasInitializer    : Initializer[TFloat32] = Initializers.select(Initializers.zeros),
                       kernelRegularizer  : Option[Nothing]       = None,
                       biasRegularizer    : Option[Nothing]       = None,
                       activityRegularizer: Option[Nothing]       = None,
                       kernelConstraint   : Option[Nothing]       = None,
                       biasConstraint     : Option[Nothing]       = None,
                     ) extends Conv2D(
  filters             = filters,
  kernelSize          = kernelSize,
  strides             = strides,
  padding             = padding,
  dataFormat          = dataFormat,
  dilationRate        = dilationRate,
  activation          = activation, // activations.get(activation),
  useBias             = useBias,
  kernelInitializer   = kernelInitializer,   // initializers.get(kernel_initializer),
  biasInitializer     = biasInitializer,     // initializers.get(bias_initializer),
  kernelRegularizer   = kernelRegularizer,   // regularizers.get(kernel_regularizer),
  biasRegularizer     = biasRegularizer,     // regularizers.get(bias_regularizer),
  activityRegularizer = activityRegularizer, // regularizers.get(activity_regularizer),
  kernelConstraint    = kernelConstraint,    // constraints.get(kernel_constraint),
  biasConstraint      = biasConstraint,      // constraints.get(bias_constraint),
  ) {

  (outputPadding, strides) match {
    case (Some((p1, p2)), (s1, s2)) if p1 >= s1 || p2 >= s2 =>
      throw new IllegalArgumentException(
        s"Strides must be greater than output padding. Received strides=$strides, outputPadding=$outputPadding."
      )
    case _ =>
  }

  override protected def build(tf: Ops, inputShape: Shape): Unit = {
    // input_shape = tf.TensorShape (input_shape)
    if (inputShape.numDimensions() != 4)
      throw new IllegalArgumentException(s"Inputs should have rank 4. Received input_shape = $inputShape")
      
    val channelAxis = getChannelAxis
    if (inputShape.size(channelAxis) == Shape.UNKNOWN_SIZE)
      throw new IllegalArgumentException(
        "The channel dimension of the inputs to `Conv2DTranspose` should be defined. " ++
        s"The input_shape received is $inputShape where axis $channelAxis (0 - based) is the channel dimension, which found to be `None`.")

    val inputDim = inputShape.size(channelAxis) // .toInt
    // XXX TODO:
//    this.input_spec = InputSpec (ndim = 4, axes = { channelAxis: inputDim })
    val kernelShape = (kernelSize._1 + filters, kernelSize._2 + inputDim)

    this.kernel = addWeight(tf,
      name        = "kernel",
      shape       = Shape.of(kernelShape._1, kernelShape._2),
      initializerName = "kernelInit",
      initializer = kernelInitializer,
      regularizer = kernelRegularizer,
      constraint  = kernelConstraint,
      trainable   = Some(true),
      dtype       = dtype
    )
    if (useBias) {
      this.bias = addWeight(tf,
        name        = "bias",
        shape       = Shape.of(filters), // (filters, ),
        initializerName = "biasInit",
        initializer = biasInitializer,
        regularizer = biasRegularizer,
        constraint  = biasConstraint,
        trainable   = Some(true),
        dtype       = dtype
      )
    } else {
      this.bias = null // None
    }
    built = true
  }

  override protected def callOne(tf: Ops, inputs0: Operand[T]): Operand[T] = ???

  //  def call (self, inputs):
  //  inputs_shape = tf.shape (inputs)
  //  batch_size = inputs_shape[0]
  //  if self.data_format == 'channels_first':
  //  h_axis, w_axis = 2, 3
  //  else:
  //  h_axis, w_axis = 1, 2
  //
  //  // Use the constant height and weight when possible.
  //  // TODO(scottzhu): Extract this into a utility function that can be applied
  //  // to all convolutional layers, which currently lost the static shape
  //  // information due to tf.shape().
  //  height, width = None, None
  //  if inputs.shape.rank is not None:
  //  dims = inputs.shape.as_list ()
  //  height = dims[h_axis]
  //  width = dims[w_axis]
  //  height = height if height is not None else inputs_shape[h_axis]
  //  width = width if width is not None else inputs_shape[w_axis]
  //
  //  kernel_h, kernel_w = self.kernel_size
  //  stride_h, stride_w = self.strides
  //
  //  if self.output_padding is None:
  //  out_pad_h = out_pad_w = None
  //  else:
  //  out_pad_h, out_pad_w = self.output_padding
  //
  //  // Infer the dynamic output shape:
  //  out_height = conv_utils.deconv_output_length (height,
  //  kernel_h,
  //  padding = self.padding,
  //  output_padding = out_pad_h,
  //  stride = stride_h,
  //  dilation = self.dilation_rate[0] )
  //  out_width = conv_utils.deconv_output_length (width,
  //  kernel_w,
  //  padding = self.padding,
  //  output_padding = out_pad_w,
  //  stride = stride_w,
  //  dilation = self.dilation_rate[1] )
  //  if self.data_format == 'channels_first':
  //  output_shape = (batch_size, self.filters, out_height, out_width)
  //  else:
  //  output_shape = (batch_size, out_height, out_width, self.filters)
  //
  //  output_shape_tensor = tf.stack (output_shape)
  //  outputs = backend.conv2d_transpose (
  //  inputs,
  //  self.kernel,
  //  output_shape_tensor,
  //  strides = self.strides,
  //  padding = self.padding,
  //  data_format = self.data_format,
  //  dilation_rate = self.dilation_rate)
  //
  //  if not tf.executing_eagerly ():
  //  // Infer the static output shape:
  //  out_shape = self.compute_output_shape (inputs.shape)
  //  outputs.set_shape (out_shape)
  //
  //  if self.use_bias:
  //  outputs = tf.nn.bias_add (
  //  outputs,
  //  self.bias,
  //  data_format = conv_utils.convert_data_format (self.data_format, ndim = 4) )
  //
  //  if self.activation is not None:
  //  return self.activation (outputs)
  //  return outputs

  override def computeOutputShape(inputShape: Shape): Shape = {
    val outputShape = inputShape.asArray()
    val (cAxis, hAxis, wAxis) = if (dataFormat == DataFormat.ChannelsFirst)
      (1, 2, 3)
    else
      (3, 1, 2)

    val (kernelH, kernelW)  = kernelSize
    val (strideH, strideW)  = strides
    val (outPadH: Option[Long], outPadW: Option[Long]) = outputPadding.unzip

    outputShape(cAxis) = filters
    outputShape(hAxis) = ConvUtils.deconvOutputLength(
      outputShape(hAxis),
      kernelH,
      padding       = padding,
      outputPadding = outPadH,
      stride        = strideH,
      dilation      = dilationRate._1
    )
    outputShape(wAxis) = ConvUtils.deconvOutputLength(
      outputShape(wAxis),
      kernelW,
      padding       = padding,
      outputPadding = outPadW,
      stride        = strideW,
      dilation      = dilationRate._2
    )
    Shape.of(outputShape: _*)
  }
}