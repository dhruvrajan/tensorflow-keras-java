package org.tensorflow.utils

import org.tensorflow.Operand
import org.tensorflow.op.{MathOps, Ops}
import org.tensorflow.op.math.Mean
import org.tensorflow.types.TInt32
import org.tensorflow.types.family.{TNumber, TType}

object MissingBits {
  implicit class MoreMathOps(private val math: MathOps) extends AnyVal {
    /** Computes the mean of elements across dimensions of a tensor.
      * Reduces `input_tensor` along the dimensions given in `axis` by computing the
      * mean of elements across the dimensions in `axis`.
      * Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
      * the entries in `axis`, which must be unique. If `keepdims` is true, the
      * reduced dimensions are retained with length 1.
      * If `axis` is None, all dimensions are reduced, and a tensor with a single
      * element is returned.
      *
      * @param inputTensor  The tensor to reduce. Should have numeric type.
      * @param axis The dimensions to reduce. If `None` (the default), reduces all
      *  dimensions. Must be in the range `[-rank(input_tensor),
      * rank(input_tensor))`.
      * @param keepDims If true, retains reduced dimensions with length 1.
      * @param name A name for the operation (optional).
      *
      * @return The reduced tensor.
      */
    def reduceMean[T <: TType, N <: TNumber](inputTensor: Operand[T], axis: Option[Operand[N]], keepDims: Boolean,
                                             name: Option[String] = None): Operand[T] = {
      val tf0 = math.ops()
      val tf  = name.fold(tf0)(tf0.withName)
      val m   = tf.math.mean(inputTensor, reductionDims(tf, inputTensor, axis), Mean.keepDims(keepDims) /*, name = name*/)
      mayReduceToScalar(keepDims, axis, m)
    }

    // Returns range(0, rank(x)) if axis is None.
    private def reductionDims[T <: TType, N <: TNumber](tf: Ops, x: Operand[T], axis: Option[Operand[N]]): Operand[_ <: TNumber] =
      axis.getOrElse {
        val rankOpt = x.shape().numDimensions()
        // Fast path: avoid creating Rank and Range ops if ndims is known.
        // Otherwise, we rely on Range and Rank to
        // do the right thing at run - time.
        val rank: Operand[TInt32] = if (rankOpt == -1) tf.rank(x) else tf.constant(rankOpt)
        tf.range[TInt32](tf.constant(0), rank, tf.constant(1))
      }

    // Sets a reduction's output shape to be a scalar if we are certain.
    private def mayReduceToScalar[T <: TType, N <: TNumber](keepDims: Boolean, axis: Option[Operand[N]],
                                                            output: Operand[T]): Operand[T] = {
      if (!hasFullyDefinedShape(output) && !keepDims && axis.isEmpty) {
        ??? // output.set_shape(())
      }
      output
    }

    // Returns true if tensor has a fully defined shape.
    private def hasFullyDefinedShape[T <: TType](tensor: Operand[T]): Boolean =
      /*isinstance(tensor, EagerTensor) ||*/ !tensor.shape.hasUnknownDimension  // XXX TODO
  }
}
