package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Slice;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TType;

import java.util.Arrays;

public class GraphModeTensorFrame<T extends TType> extends TensorFrame<T> implements GraphLoader<T>, AutoCloseable {
    private final Class<T> dtype;

    private final Tensor[] dataTensors;
    private Placeholder<T>[] dataPlaceholders;
    private Slice<T>[] batchOperands;

    private Placeholder<TInt64>[] batchStart;
    private Placeholder<TInt64>[] batchSize;

    private boolean built = false;

//    @SafeVarargs
    public GraphModeTensorFrame(Class<T> dtype, Tensor firstTensor, Tensor... tensors) {
        this.dtype = dtype;

        // Check first dimension matches
        long matchDim = firstTensor.shape().size(0);

        for (Tensor t : tensors) {
            if (t.shape().size(0) != matchDim) {
                throw new IllegalArgumentException(
                        "All dataTensors in a tensor frame must have equal first dimension.");
            }
        }

        // Record Tensor Objects
        this.dataTensors = new Tensor[tensors.length + 1];
        this.dataTensors[0] = firstTensor;
        System.arraycopy(tensors, 0, this.dataTensors, 1, tensors.length);
    }

    public int length() {
        return this.dataTensors.length;
    }

    public long size() {
        return this.dataTensors[0].shape().size(0);
    }

    public void build(Ops tf) {
        // Create Placeholders (will be filled by dataTensors before graph is run)
        this.dataPlaceholders = new Placeholder[this.length()];
        for (int i = 0; i < this.length(); ++i) {
            this.dataPlaceholders[i] =
                    tf.placeholder(this.dtype, Placeholder.shape(this.dataTensors[i].shape()));
        }

        // Placeholder representing batch start and size selectors.
        this.batchStart = new Placeholder[this.length()];
        this.batchSize  = new Placeholder[this.length()];

        for (int i = 0; i < this.length(); i++) {
            batchStart[i] = tf.placeholder(TInt64.class, Placeholder.shape(Shape.of(this.dataTensors[i].shape().numDimensions())));
            batchSize[i]  = tf.placeholder(TInt64.class, Placeholder.shape(Shape.of(this.dataTensors[i].shape().numDimensions())));
        }


        // Create batch slice operands
        this.batchOperands = new Slice[this.length()];
        for (int i = 0; i < this.length(); i++) {
            batchOperands[i] = tf.slice(dataPlaceholders[i], batchStart[i], batchSize[i]);
        }

        this.built = true;
    }

    @Override
    public Session.Runner feedSessionRunner(Session.Runner runner, long batch) {
        // Feed Data Tensors
        for (int i = 0; i < dataPlaceholders.length; i++) {
            runner.feed(dataPlaceholders[i].asOutput(), dataTensors[i]);
        }

        // Feed Batch Selectors
        for (int i = 0; i < this.length(); i++) {
            long[] start = new long[dataTensors[i].shape().numDimensions()];
            Arrays.fill(start, 0);
            start[0] = batch * this.batchSize();

            long[] size = new long[dataTensors[i].shape().numDimensions()];
            Arrays.fill(size, -1);
            size[0] = this.batchSize();

            runner.feed(this.batchStart[i].asOutput(), TInt64.vectorOf(start));
            runner.feed(this.batchSize [i].asOutput(), TInt64.vectorOf(size));
        }

        return runner;
    }

    public boolean isBuilt() {
        return built;
    }

    public Tensor[] getDataTensors() {
        return dataTensors;
    }

    public Placeholder<T>[] getDataPlaceholders() {
        return dataPlaceholders;
    }

    public Operand<T>[] getBatchOperands() {
        return batchOperands;
    }

    /**
     * Utility to construct a Shape from a long[]
     */
    private static Shape getShape(long... dims) {
        assert dims.length > 0;
        return Shape.of(dims);
    }

    @Override
    public void close() {
        for (Tensor tensor : this.dataTensors) {
            tensor.close();
        }
    }
}
