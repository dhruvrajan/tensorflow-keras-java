package io.gitlab.tensorflow.keras.data;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;

import java.util.Arrays;
import java.util.Iterator;

/**
 * Class representing a 'split' (e.g. train, val, test) of data.
 * A split contains a data tensor (X) and label tensor(y).
 *
 * The first dimension of these tensors denotes batch size.
 *
 * @param <T> Numeric dtype of tensor values.
 */
public class TensorSplit<T extends Number> {
    // Dataset Tensors
    private Tensor<T> X;
    private Tensor<T> y;

    // Dataset Operands
    private Operand<T> XOp;
    private Operand<T> yOp;

    // Batch Operands
    private Operand<T>[] XBatches;
    private Operand<T>[] yBatches;

    // Dataset Size
    private long N;

    private boolean built = false;
    private Class<T> dtype;

    public TensorSplit(Tensor<T> X, Tensor<T> y, Class<T> dtype) {
        // Sizes of data and labels must be equal
        assert X.shape()[0] == y.shape()[0];
        this.N = X.shape()[0];

        this.X = X;
        this.y = y;
        this.dtype = dtype;
    }


    public Tensor<T> getX() {
        return X;
    }

    public Tensor<T> getY() {
        return y;
    }

    public long getN() {
        return N;
    }

    public Operand<T> getXOp() {
        return XOp;
    }

    public Operand<T> getyOp() {
        return yOp;
    }

    public long getSize() {
        return this.N;
    }


    /**
     * Build the data loader graph. Populates batch operands as tf.slice objects
     * @param tf Tensorflow Ops object
     * @param batchSize Batch size for split. Must evenly divide the dataset size.
     */
    @SuppressWarnings("unchecked")
    public void build(Ops tf, int batchSize) {
        assert X.shape()[0] % batchSize == 0;
        assert !built;

        this.XOp = (Operand<T>) tf.variable(getShape(X.shape()), dtype);
        this.yOp = (Operand<T>) tf.variable(getShape(y.shape()), dtype);

        this.XBatches = new Operand[(int) this.N / batchSize];
        this.yBatches = new Operand[(int) this.N / batchSize];
        for (int i = 0; i < XBatches.length; i++) {
            XBatches[i] = tf.slice(XOp,
                    tf.constant(getBatchStartSelector(i * batchSize, X.numDimensions())),
                    tf.constant(getBatchSizeSelector(batchSize, X.numDimensions())));
            yBatches[i] = tf.slice(yOp,
                    tf.constant(getBatchStartSelector(i * batchSize, y.numDimensions())),
                    tf.constant(getBatchSizeSelector(batchSize, y.numDimensions())));
        }

        built = true;
        assert XBatches.length == yBatches.length;
    }

    /**
     * Get an iterator over batches drawn from X.
     */
    public Operand<T>[] XBatches() {
        assert built;
        return XBatches;
    }

    /**
     * Get an iterator over batches drawn from y.
     */
    public Operand<T>[] yBatches() {
        assert built;
        return yBatches;
    }

    /**
     * Get an iterator over batches drawn from X.
     */
    public Iterator<Operand<T>> XBatchIterator() {
        assert built;
        return Arrays.stream(XBatches).iterator();

    }

    /**
     * Get an iterator over batches drawn from y.
     */
    public Iterator<Operand<T>> yBatchIterator() {
        assert built;

        return Arrays.stream(yBatches).iterator();
    }

    public int numBatches() {
        return XBatches.length;
    }

    /**
     * Build a long[length] filled with default_, with val at position pos
     */
    private static long[] batchSelector(int length, int pos, long val, long default_) {

        long[] arr = new long[length];
        Arrays.fill(arr, default_);
        arr[pos] = val;
        return arr;
    }

    /**
     * Size selector for tf.slice
     */
    private static long[] getBatchSizeSelector(int batchSize, int dimensions) {
        return batchSelector(dimensions, 0, batchSize, -1);
    }

    /**
     * Start selector for tf.slice
     */
    private static long[] getBatchStartSelector(int target, int dimensions) {
        return batchSelector(dimensions, 0, target, 0);
    }

    /**
     * Utility to construct a Shape from a long[]
     */
    private static Shape getShape(long... dims) {
        assert dims.length > 0;

        long head = dims[0];
        long[] tail = new long[dims.length - 1];
        System.arraycopy(dims, 1, tail, 0, dims.length - 1);

        return Shape.make(head, tail);
    }
}
