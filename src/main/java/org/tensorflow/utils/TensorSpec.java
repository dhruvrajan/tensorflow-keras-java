package org.tensorflow.utils;

import org.tensorflow.DataType;
import org.tensorflow.Shape;

/**
 * Describes a tf.Tensor
 *
 * <p>Class to represent metadata describing the `tf.Tensor` objects accepted or returned by some
 * TensorFlow APIs.
 */
public class TensorSpec {

  private Shape shape;
  private DataType dtype;
  private String name;

  /**
   * Creates a `TensorSpec`.
   *
   * @param shape The shape of the tensor.
   * @param dtype The type of the tensor values.
   */
  private TensorSpec(Shape shape, DataType dtype) {
    this.shape = shape;
    this.dtype = dtype;
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
