package org.tensorflow.keras.utilss

import org.tensorflow.Operand
import org.tensorflow.keras.layers.Conv
import org.tensorflow.keras.layers.Conv.Padding
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.family.TType
import org.tensorflow.utils.Implicits.ShapeOps

object ConvUtils {
  /** Determines output length of a convolution given input length. */
  def convOutputLength(inputLength: Long, filterSize: Long, padding: Conv.Padding, stride: Long,
                       dilation: Long = 1): Long = {
    val dilatedFilterSize = filterSize + (filterSize - 1) * (dilation - 1)
    val outputLength = padding match {
      case Padding.Same | Padding.Causal  => inputLength
      case Padding.Valid                  => inputLength - dilatedFilterSize + 1
      case Padding.Full                   => inputLength + dilatedFilterSize - 1
    }
    (outputLength + stride - 1) / stride
  }

  /** Returns `unsqueeze_batch(op(squeeze_batch(inp)))`.
    * Where `squeeze_batch` reshapes `inp` to shape
    * `[prod(inp.shape[:-inner_rank])] + inp.shape[-inner_rank:]`
    * and `unsqueeze_batch` does the reverse reshape but on the output.
    *
    * @param inp  A tensor with dims `batch_shape + inner_shape` where `inner_shape`is length `innerRank`.
    * @param op   A function that takes a single input tensor and returns a single.
    *             output tensor.
    * @return `unsqueeze_batch_op(squeeze_batch(inp))`
    */
  def squeezeBatchDims[T <: TType](tf: Ops, inp: Operand[T], op: Operand[T] => Operand[T], innerRank: Int): Operand[T] = {
//    with tf.name_scope('squeeze_batch_dims'):
    val shape   = inp.shape

    var innerShape = shape.takeRight(innerRank)
    if (!innerShape.isFullyDefined)
      innerShape = tf.shape(inp).shape().takeRight(innerRank) // XXX TODO correct?

    var batchShape: Shape = shape.dropRight(innerRank)
    if (!batchShape.isFullyDefined)
      batchShape = tf.shape(inp).shape().dropRight(innerRank) // XXX TODO correct?

    val inpReshaped: Operand[T] = /*if (isinstance(innerShape, tf.TensorShape))*/
//      tf.reshape(inp, -1L +: innerShape.asArray())
    /*else*/
      ??? // tf.reshape(inp, tf.concat(([-1], innerShape), axis=-1))

    val outReshaped = op(inpReshaped)

    var outInnerShape: Shape = outReshaped.shape.takeRight(innerRank)
    if (!outInnerShape.isFullyDefined)
      outInnerShape = tf.shape(outReshaped).shape().takeRight(innerRank)  // XXX TODO correct?

    val out = tf.reshape(
      outReshaped, tf.constant(batchShape append outInnerShape)) // XXX TODO correct? // tf.concat((batchShape, outInnerShape), /*axis=*/-1))

    ??? // out.set_shape(inp.shape.dropRight(innerRank) + out.shape.takeRight(innerRank))
    out
  }
}
