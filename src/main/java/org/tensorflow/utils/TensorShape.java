package org.tensorflow.utils;

import org.tensorflow.Shape;

import java.util.Arrays;

/**
 * Represents the shape of a `Tensor`.
 *
 * <p>A `TensorShape` represents a possibly-partial shape specification for a `Tensor`. It may be
 * one of the following:
 *
 * <p>Fully-known shape: has a known number of dimensions and known size for each dimension.
 *
 * <p>Partially-known shape: has a known number of dimensions, and an unknown size in all
 * dimensions.
 *
 * <p>Unknown shape: has an unknown number of dimensions, and an unknown size in all dimensions.
 */
public class TensorShape {
  private long[] dims;

  /**
   * Creates a new `TensorShape` with the given dimensions.
   *
   * @param firstDimension The size of the first dimension
   * @param dims The sizes of the remaining dimensions
   */
  public TensorShape(long firstDimension, long... dims) {
    this.dims = new long[dims.length + 1];
    this.dims[0] = firstDimension;
    System.arraycopy(dims, 0, this.dims, 1, dims.length);
  }

  /** Creates a `TensorShape` object from a TensorFlow `Shape` object */
  public TensorShape(Shape shape) {
    this.dims = dimsFromShape(shape);
  }

  /** Returns the rank of this shape. */
  public int rank() {
    return dims.length;
  }

  /** Returns the array of dimensions representing this shape. */
  public long[] dims() {
    return this.dims;
  }

  /**
   * Returns the value of a dimension
   *
   * @param i The index at which to retrieve a dimension.
   * @return The size of dimension i
   */
  public long get(int i) {
    return this.dims[i];
  }

  /**
   * Test whether dimension i in this shape is known
   *
   * @param i Target dimension to test
   * @return Whether dimension i is unknown (equal to -1)
   */
  public boolean isKnown(int i) {
    return dims[i] != -1;
  }

  /**
   * Throw an exception if dimension i is unknown.
   *
   * @param i Target dimension to test
   * @throws IllegalStateException if dimension i is unknown
   */
  public void assertKnown(int i) {
    if (!isKnown(i)) {
      throw new IllegalStateException("Dimension " + i + " in shape needs to be known.");
    }
  }

  /**
   * Replace dimension i with a new dimension size.
   *
   * @param i The target dimension to change.
   * @param dim The new dimension size.
   * @return The new changed TensorShape
   */
  public TensorShape replace(int i, long dim) {
    dims[i] = dim;
    return this;
  }

  /**
   * Replace the last dimension with a new dimension size.
   *
   * @param dim New size for the last dimensions
   * @return The new changed TensorShape
   */
  public TensorShape replaceLast(long dim) {
    return replace(this.dims.length - 1, dim);
  }

  /**
   * Replace the first dimension with a new dimension size.
   *
   * @param dim New size for first dimension
   * @return The new changed TensorShape.
   */
  public TensorShape replaceFirst(long dim) {
    return replace(0, dim);
  }

  /**
   * Get the size of a target dimension.
   *
   * @param i Target dimension.
   * @return The size of dimension i
   */
  public long size(int i) {
    return this.dims[i];
  }

  /**
   * Augment this TensorShape by appending more dimensions to it.
   *
   * @param dims The new dimensions to incorporate
   * @return The new changed TensorShape
   */
  public TensorShape concatenate(long... dims) {
    this.dims = concatenate(this.dims, dims);
    return this;
  }

  @Override
  public boolean equals(Object other) {
    if (other == this) return true;
    else if (!(other instanceof TensorShape)) return false;
    return Arrays.equals(((TensorShape) other).dims, dims);
  }

  private static long[] dimsFromShape(Shape shape) {
    long[] dims = new long[shape.numDimensions()];
    for (int i = 0; i < shape.numDimensions(); i++) {
      dims[i] = shape.size(i);
    }
    return dims;
  }

  private static long[] concatenate(long[] first, long... last) {
    long[] dims = new long[first.length + last.length];
    System.arraycopy(first, 0, dims, 0, first.length);
    System.arraycopy(last, 0, dims, first.length, last.length);
    return dims;
  }

  private static long head(long... dims) {
    return dims[0];
  }

  private static long[] tail(long... dims) {
    return Arrays.copyOfRange(dims, 1, dims.length);
  }

  public Shape toShape() {
    return Shape.make(head(dims), tail(dims));
  }
}
