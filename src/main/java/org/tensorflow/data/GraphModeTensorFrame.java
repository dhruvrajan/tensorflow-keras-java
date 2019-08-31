package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.utils.Pair;

import java.util.Arrays;
import java.util.Iterator;
import java.util.stream.Collectors;

public class GraphModeTensorFrame<T> extends TensorFrame<T> implements GraphLoader<T> {
    private Class<T> dtype;
    private Tensor<T>[] tensors;

    @SafeVarargs
    public GraphModeTensorFrame(Class<T> dtype, Tensor<T> firstTensor, Tensor<T>... tensors) {
        this.dtype = dtype;

        // Check first dimension matches
        long matchDim = firstTensor.shape()[0];
        for (Tensor<T> t: tensors) {
            if (t.shape()[0] != matchDim) {
                throw new IllegalArgumentException("All tensors in a tensor frame must have equal first dimension.");
            }
        }

        this.tensors = new Tensor[tensors.length + 1];
        this.tensors[0] = firstTensor;
        System.arraycopy(tensors, 0, this.tensors, 1, tensors.length);
    }

    public long size() {
        return this.tensors[0].shape()[0];
    }

    /** Utility to construct a Shape from a long[] */
    private static Shape getShape(long... dims) {
        assert dims.length > 0;

        long head = dims[0];
        long[] tail = new long[dims.length - 1];
        System.arraycopy(dims, 1, tail, 0, dims.length - 1);

        return Shape.make(head, tail);
    }

    private Operand<T>[] getBatch(Ops tf, long i) {
        Operand<T>[] ops = new Operand[this.tensors.length];
        return Arrays.stream(tensors)
                .map(tensor -> {
                    Operand<T> variable = tf.variable(getShape(tensor.shape()), dtype);
                    return tf.slice(variable,
                            tf.constant(getBatchStartSelector((int) i * batchSize, tensor.numDimensions())),
                            tf.constant(getBatchSizeSelector(batchSize, tensor.numDimensions())));

                }).collect(Collectors.toList()).toArray(ops);
    }

    /** Build a long[length] filled with default_, with val at position pos */
    private static long[] batchSelector(int length, int pos, long val, long default_) {

        long[] arr = new long[length];
        Arrays.fill(arr, default_);
        arr[pos] = val;
        return arr;
    }

    /** Size selector for tf.slice */
    private static long[] getBatchSizeSelector(long batchSize, int dimensions) {
        return batchSelector(dimensions, 0, batchSize, -1);
    }

    /** Start selector for tf.slice */
    private static long[] getBatchStartSelector(long target, int dimensions) {
        return batchSelector(dimensions, 0, target, 0);
    }

    @Override
    public Iterator<Pair<Tensor<T>[], Operand<T>[]>> getBatchTensorsAndOps(Ops tf) {
        return new Iterator<Pair<Tensor<T>[], Operand<T>[]>>() {
            long batchIndex = 0;
            @Override
            public boolean hasNext() {
                return batchIndex < numBatches();
            }

            @Override
            public Pair<Tensor<T>[], Operand<T>[]> next() {
                Operand<T>[] batchOps = getBatch(tf, batchIndex);
                batchIndex++;
                return new Pair<>(tensors, batchOps);
            }
        };
    }
}
