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

  public boolean isKnown(int i) {
    return dims[i] != -1;
  }

  public void assertKnown(int i) {
    if (!isKnown(i)) {
      throw new IllegalStateException("Dimension " + i + " in shape needs to be known.");
    }
  }

  public TensorShape replaceLast(long dim) {
    return replace(this.dims.length - 1, dim);
  }

  public long size(int i) {
    return this.dims[i];
  }

  public int numDimensions() {
    return this.dims.length;
  }

  public TensorShape replace(int i, long dim) {
    dims[i] = dim;
    return this;
  }

  public TensorShape concatenate(long dim) {
    this.dims = concatenate(this.dims, dim);
    return this;
  }

  public static long[] dimsFromShape(Shape shape) {
    long[] dims = new long[shape.numDimensions()];
    for (int i = 0; i < shape.numDimensions(); i++) {
      dims[i] = shape.size(i);
    }
    return dims;
  }

  public static long[] concatenate(long[] first, long last) {
    long[] dims = new long[first.length + 1];
    System.arraycopy(first, 0, dims, 0, first.length);
    dims[dims.length - 1] = last;
    return dims;
  }

  public static long head(long... dims) {
    return dims[0];
  }

  public static long[] tail(long... dims) {
    return Arrays.copyOfRange(dims, 1, dims.length);
  }

  public Shape toShape() {
    return Shape.make(head(dims), tail(dims));
  }
}
