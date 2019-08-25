package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.utils.Pair;

import java.util.Collection;
import java.util.Iterator;

public class GraphTensorDataset<T> extends Dataset<T> {
    NdArray<T>[] tensors;

    @Override
    public Dataset<T> batch(int batchSize, boolean dropRemainder) {
        return null;
    }

    @Override
    public Iterator<Pair<Integer, Collection<T>>> enumerate(int start) {
        return null;
    }

    @Override
    public Iterator<Collection<Operand<T>>> iterator() {
        return null;
    }
}
