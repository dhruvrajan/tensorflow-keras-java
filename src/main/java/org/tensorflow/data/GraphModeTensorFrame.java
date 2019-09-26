package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.stream.Collectors;

public class GraphModeTensorFrame<T> extends TensorFrame<T> {
  private Class<T> dtype;

  private Tensor<T>[] dataTensors;
  private Placeholder<T>[] dataPlaceholders;
  private Operand<T>[] batchOperands;

  private Placeholder<Integer> batchIndexPlaceholder;
  private boolean built = false;

  @SafeVarargs
  public GraphModeTensorFrame(Class<T> dtype, Tensor<T> firstTensor, Tensor<T>... tensors) {
    this.dtype = dtype;

    // Check first dimension matches
    long matchDim = firstTensor.shape()[0];

    for (Tensor<T> t : tensors) {
      if (t.shape()[0] != matchDim) {
        throw new IllegalArgumentException(
            "All dataTensors in a tensor frame must have equal first dimension.");
      }
    }

    // Record Tensor Objects
    this.dataTensors = (Tensor<T>[]) new Tensor[tensors.length + 1];
    this.dataTensors[0] = firstTensor;
    System.arraycopy(tensors, 0, this.dataTensors, 1, tensors.length);
  }

  public long size() {
    return this.dataTensors[0].shape()[0];
  }

  /** Utility to construct a Shape from a long[] */
  private static Shape getShape(long... dims) {
    assert dims.length > 0;

    long head = dims[0];
    long[] tail = new long[dims.length - 1];
    System.arraycopy(dims, 1, tail, 0, dims.length - 1);

    return Shape.make(head, tail);
  }

  private Operand<T>[] makeBatchOps(Ops tf) {
    long[] startSelector = new long[this.dataTensors[0].numDimensions()];
    long[] sizeSelector = new long[this.dataTensors[0].numDimensions()];

    Arrays.fill(startSelector, 0);
    Arrays.fill(sizeSelector, -1);

    sizeSelector[0] = batchSize;

    Operand<Long> sizeSelectorOp = tf.constant(sizeSelector);
    Operand<Long>

    for (Placeholder<T> placeholder : this.dataPlaceholders) {

    }
  }

  private Operand<T>[] getBatch(Ops tf, long i) {
    Operand<T>[] ops = new Operand[this.dataTensors.length];


    return Arrays.stream(dataPlaceholders)
        .map(
            placeholder -> {
              Shape shape = placeholder.asOutput().shape();
              return tf.slice(
                  placeholder,
                  tf.constant(getBatchStartSelector((int) i * batchSize, shape.numDimensions())),
                  tf.constant(getBatchSizeSelector(batchSize, shape.numDimensions())));
            })
        .collect(Collectors.toList())
        .toArray(ops);
  }

  /** Build a long[length] filled with default_, with val at position pos */
  private static long[] batchSelector(int length, int pos, long val, long default_) {

    long[] arr = new long[length];
    Arrays.fill(arr, default_);
    arr[pos] = val;
    return arr;
  }

  /** Size selector for tf.slice */
  private static Operand<T> getBatchSizeSelector(long batchSize, int dimensions) {
    return batchSelector(dimensions, 0, batchSize, -1);
  }

  /** Start selector for tf.slice */
  private static long[] getBatchStartSelector(long target, int dimensions) {
    return batchSelector(dimensions, 0, target, 0);
  }

  public GraphModeTensorFrame<T> build(Ops tf) {
    // Create Placeholders (will be filled by dataTensors before graph is run)
    this.dataPlaceholders = new Placeholder[this.dataTensors.length];
    for (int i = 0; i < this.dataPlaceholders.length; ++i) {
      this.dataPlaceholders[i] =
          tf.placeholder(this.dtype, Placeholder.shape(getShape(this.dataTensors[i].shape())));
    }

    // Placeholder representing batch index
    this.batchIndexPlaceholder = tf.placeholder(Integer.class, Placeholder.shape(Shape.make(1)));


    this.built = true;
    return this;
  }

  public boolean isBuilt() {
    return built;
  }

  public Tensor<T>[] getDataTensors() {
    return dataTensors;
  }

  public Placeholder<T>[] getDataPlaceholders() {
    return dataPlaceholders;
  }

  public Operand<T>[] getBatchOperands(Ops tf) {
    return batchOperands;
  }


//  public Iterator<Operand<T>[]> getBatchOps(Ops tf) {
//    if (!built) throw new IllegalStateException("Must build tensorframe before getting batches");
//
//    return new Iterator<>() {
//      int batchIndex = 0;
//
//      @Override
//      public boolean hasNext() {
//        return batchIndex < numBatches();
//      }
//
//      @Override
//      public Operand<T>[] next() {
//        return batchOperands[batchIndex];
//      }
//    };
//  }
}
