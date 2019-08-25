package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.utils.Pair;
import org.tensorflow.op.Ops;

import java.util.Collection;
import java.util.Iterator;

public abstract class TensorFrame<T> {
    /**
     * Splits a TensorFrame into a train frame and test frame, using `testSize` as the fraction
     * to keep in the test frame.
     * @return A pair (train, test) containing the new frames.
     */
    public abstract Pair<TensorFrame<T>, TensorFrame<T>> trainTestSplit(double testSize);

    /**
     * Constructs an iterator over batches of size `batchSize`.
     * @param tf
     * @param batchSize
     * @return
     */
    public abstract Iterator<Collection<Operand<T>>> batchIterator(Ops tf, int batchSize);
    public abstract long size();
}
