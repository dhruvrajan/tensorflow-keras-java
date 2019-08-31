package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.op.Ops;
import org.tensorflow.utils.Pair;

import java.util.Collection;
import java.util.Iterator;

public abstract class TensorFrame<T> implements Dataset<T> {
    protected long batchSize = 1;
    protected boolean dropRemainder = false;

    /* Override functions from Dataset<T> */
    @Override
    public Dataset<T> batch(long batchSize, boolean dropRemainder) {
        this.batchSize = batchSize;
        this.dropRemainder = dropRemainder;
        return this;
    }
}