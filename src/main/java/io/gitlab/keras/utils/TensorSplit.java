package io.gitlab.keras.utils;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;

import java.util.Arrays;
import java.util.Iterator;

class TensorSplit<T> {
    public Tensor<T> X;
    public Tensor<T> y;

    private long N;

    public Operand<T> XOp;
    public Operand<T> yOp;

    Operand<T>[] XBatches;
    Operand<T>[] yBatches;

    private boolean built = false;

    private Class<T> dtype;

    public TensorSplit(Tensor<T> X, Tensor<T> y, Class<T> dtype) {
        // Batch Dimensions Equal
        assert X.shape()[0] == y.shape()[0];
        this.X = X;
        this.y = y;

        this.dtype = dtype;

        this.N = X.shape()[0];
    }

    @SuppressWarnings("unchecked")
    public void build(Ops tf, int batchSize) {
        assert X.shape()[0] % batchSize == 0;
        assert !built;

        this.XOp = (Operand<T>) tf.variable(getShape(X.shape()), dtype);
        this.yOp = (Operand<T>) tf.variable(getShape(y.shape()), dtype);

        this.XBatches = new Operand[(int) X.shape()[0] / batchSize];
        this.yBatches = new Operand[(int) y.shape()[0] / batchSize];
        for (int i = 0; i < XBatches.length; i++) {
            XBatches[i] = tf.slice(XOp,
                    tf.constant(getBatchStartSelector(i * batchSize, X.numDimensions())),
                    tf.constant(getBatchSizeSelector(batchSize, X.numDimensions())));
            yBatches[i] = tf.slice(yOp,
                    tf.constant(getBatchStartSelector(i * batchSize, y.numDimensions())),
                    tf.constant(getBatchSizeSelector(batchSize, y.numDimensions())));
        }

        built = true;
    }

    public Iterator<Operand<T>> XBatchIterator() {
        assert built;
        return Arrays.stream(XBatches).iterator();

    }

    public Iterator<Operand<T>> yBatchIterator() {
        assert built;

        return Arrays.stream(yBatches).iterator();
    }

    private static long[] batchSelector(int length, int pos, long val, long default_) {
        long[] arr = new long[length];
        Arrays.fill(arr, default_);
        arr[pos] = val;
        return arr;
    }

    private static long[] getBatchSizeSelector(int batchSize, int dimensions) {
        return batchSelector(dimensions, 0, batchSize, -1);
    }

    private static long[] getBatchStartSelector(int target, int dimensions) {
        return batchSelector(dimensions, 0, target, 0);
    }


    private static Shape getShape(long... dims) {
        assert dims.length > 0;

        long head = dims[0];
        long[] tail = new long[dims.length - 1];
        System.arraycopy(dims, 1, tail, 0, dims.length - 1);

        return Shape.make(head, tail);
    }
}
