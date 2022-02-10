package org.tensorflow.keras.layers

import org.tensorflow.Operand
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.TInt64
import org.tensorflow.types.family.TNumber
import org.tensorflow.utils.TensorShape

// based on https://github.com/keras-team/keras/blob/master/keras/layers/reshaping/reshape.py

/** Layer that reshapes inputs into the given shape.
  *
  * Input shape:
  *   Arbitrary, although all dimensions in the input shape must be known/fixed.
  *   Use the keyword argument `input_shape` (tuple of integers, does not include
  *   the samples/batch size axis) when using this layer as the first layer
  *   in a model.
  *
  * Output shape:
  *   `(batch_size,) + target_shape`
  */
class Reshape[T <: TNumber](targetShape: Shape) extends Layer[T](1) {
  override protected def build(tf: Ops, inputShape: Shape): Unit = ()

  override def computeOutputShape(inputShape: Shape): Shape = {
    val outputShape0 = inputShape.size(0)
    val outputShapeR: Array[Long] = if (inputShape.hasUnknownDimension) {
      targetShape.asArray()
    } else {
      fixUnknownDimension(inputShape.tail(), targetShape).asArray()
    }
    new TensorShape(outputShape0, outputShapeR: _*).toShape
  }

  /** Finds and replaces a missing dimension in an output shape.
    * This is a near direct port of the internal Numpy function
    * `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`
    *
    * @throws IllegalArgumentException If the total array size of the output_shape is
    *   different than the input_shape, or more than one unknown dimension
    *   is specified.
    *
    * @param inputShape   Shape of array being reshaped
    * @param outputShape  Desired shape of the array with at most a single -1 which
    *     indicates a dimension that should be derived from the input shape.
    * @return The new output shape with a -1 replaced with its computed value.
    */
  private def fixUnknownDimension(inputShape: Shape, outputShape: Shape): Shape = {
    def error() = throw new IllegalArgumentException(
      s"total size of new array must be unchanged, inputShape = $inputShape, outputShape = $outputShape"
    )

    val inputArr  = inputShape  .asArray()
    val outputArr = outputShape .asArray()
    val original  = inputArr  .product
    val known0    = outputArr .product

    if (outputShape.hasUnknownDimension) {
      val known     = -known0
      if (known == 0L || (original % known) != 0) error()
      val unknown   = outputArr.indexOf(Shape.UNKNOWN_SIZE)
      outputArr(unknown) = original / known
      Shape.of(outputArr: _*)

    } else {
      val known = known0
      if (original != known) error()
      outputShape
    }
  }

  override protected def call(tf: Ops, inputs: Seq[Operand[T]], training: Option[Boolean]): Operand[T] =
    callOne(tf, inputs.head)

  private def callOne(tf: Ops, inputs: Operand[T]): Operand[T] = {
    // XXX TODO why is the default TInt32, and should we use that?
    val newShape  = tf.shape.append(tf.shape.head(tf.shape(inputs, classOf[TInt64]), classOf[TInt64]), tf.constant(targetShape))
    val result    = tf.reshape(inputs, newShape)
    if (!tf.scope().env().isEager) {
      // Set the static shape for the result since it might lost during array_ops
      // reshape, eg, some `None` dim in the result could be inferred.
      ??? // result.set_shape(computeOutputShape(inputs.shape))
    }
    result
  }
}
