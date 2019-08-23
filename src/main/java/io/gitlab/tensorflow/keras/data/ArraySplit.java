package io.gitlab.tensorflow.keras.data;

import io.gitlab.tensorflow.keras.utils.Pair;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

import java.util.Iterator;

public class ArraySplit<T> extends KerasSplit<T> {
    float[][] X;
    float[][] y;

    @Override
    public Pair<KerasSplit<T>, KerasSplit<T>> split(double splitBy) {
        return null;
    }

    @Override
    public Pair<KerasSplit<T>, KerasSplit<T>> split(int splitIndex) {
        return null;
    }

    @Override
    public Iterator<Operand<T>> batchIterator(Ops tf, int batchSize) {
        return null;
    }

    @Override
    public void shuffle() {

    }

    @Override
    public long size() {
        return 0;
    }
}
