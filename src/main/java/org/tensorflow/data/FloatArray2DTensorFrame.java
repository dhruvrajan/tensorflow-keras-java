package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.op.Ops;
import org.tensorflow.utils.Pair;

import java.util.Collection;
import java.util.Iterator;

public class FloatArray2DTensorFrame<T> extends TensorFrame<T> {
    private float[][] X;
    private float[][] y;


    @Override
    public Pair<TensorFrame<T>, TensorFrame<T>> trainTestSplit(double testSize) {
        return null;
    }

    @Override
    public Iterator<Collection<Operand<T>>> batchIterator(Ops tf, int batchSize) {
        return null;
    }

    @Override
    public long size() {
        return 0;
    }

    
}
