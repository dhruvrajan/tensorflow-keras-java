package org.tensorflow.utils;

import org.tensorflow.ndarray.Shape;
import org.tensorflow.proto.framework.DataType;

/**
 * Describes a tf.Tensor
 *
 * <p>Class to represent metadata describing the `tf.Tensor` objects accepted or returned by some
 * TensorFlow APIs.
 */
public class TensorSpec {

  private final Shape shape;
  private final DataType dtype;
  private final String name;

  /**
   * Creates a `TensorSpec`.
   *
   * @param shape The shape of the tensor.
   * @param dtype The type of the tensor values.
   */
  private TensorSpec(Shape shape, DataType dtype) {
    this.shape = shape;
    this.dtype = dtype;
    this.name  = null;
  }

  /**
   * Creates a named `TensorSpec`.
   *
   * @param shape The shape of the tensor.
   * @param dtype The type of the tensor values.
   * @param name Name for the tensor.
   */
  private TensorSpec(Shape shape, DataType dtype, String name) {
    this.shape = shape;
    this.dtype = dtype;
    this.name = name;
  }
}
