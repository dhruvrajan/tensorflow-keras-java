package io.gitlab.keras.data;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * Class representing a 'split' (e.g. train, val, test) of data.
 * A split contains a data tensor (X) and label tensor(y).
 *
 * The first dimension of these tensors denotes batch size.
 *
 * @param <T> Numeric dtype of tensor values.
 */
public class CompactTensorSplit<T extends Number> {
    // Dataset Tensors
    private Tensor<T> X;
    private Tensor<T> y;

    // Dataset Operands
    private Operand<T> XOp;
    private Operand<T> yOp;

    // Batch Operands
    private Operand<T> XBatch;
    private Operand<T> yBatch;

    // Dataset Size
    private long N;

    private boolean built = false;
    private int batchSize;
    private Class<T> dtype;

    public CompactTensorSplit(Tensor<T> X, Tensor<T> y, Class<T> dtype) {
        // Sizes of data and labels must be equal
        assert X.shape()[0] == y.shape()[0];
        this.N = X.shape()[0];

        this.X = X;
        this.y = y;
        this.dtype = dtype;
    }

    public long numBatches() {
        return N /((long) batchSize);
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
    public void build(Ops tf, int batchSize) {
        assert X.shape()[0] % batchSize == 0;
        assert !built;

        this.XOp = tf.variable(getShape(X.shape()), dtype);
        this.yOp = tf.variable(getShape(y.shape()), dtype);

        this.batchSize = batchSize;
        built = true;
    }

    public List<Operand<T>> loadBatch(Ops tf, int batch) {
        // Assign batch operands to be run in session.run
        return Arrays.asList(
             // Assign X Batch
             tf.assign(this.XBatch, tf.slice(this.XOp,
                     tf.constant(getBatchStartSelector(batch * batchSize, X.numDimensions())),
                     tf.constant(getBatchSizeSelector(batchSize, X.numDimensions())))),

             // Assign y Batch
             tf.assign(this.yBatch, tf.slice(this.XOp,
                     tf.constant(getBatchStartSelector(batch * batchSize, y.numDimensions())),
                     tf.constant(getBatchSizeSelector(batchSize, y.numDimensions()))))
        );
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
