package org.tensorflow.keras.data;

import org.tensorflow.keras.utils.Pair;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

import java.util.Iterator;

public class KerasDataset<T> {
    private KerasSplit<T> train;
    private KerasSplit<T> val;
    private KerasSplit<T> test;


    public KerasDataset(KerasSplit<T> data) {
        this.train = data;
    }

    public KerasDataset(KerasSplit<T> data, double testSplit) {
        this.train = data;
        this.makeTestSplit(testSplit);
    }

    public KerasDataset(KerasSplit<T> data, double valSplit, double testSplit) {
        this.train = data;
        long size = this.train.size();

        this.makeValSplit(valSplit);
        this.makeTestSplit(size * testSplit / this.train.size());
    }

    public KerasDataset(KerasSplit<T> train, KerasSplit<T> test) {
        this.train = train;
        this.test = test;
    }

    public KerasDataset(KerasSplit<T> train, KerasSplit<T> val, KerasSplit<T> test) {
        this.train = train;
        this.val = val;
        this.test = test;
    }

    public Iterator<Operand<T>> trainIterator(Ops tf) {
        return this.train.batchIterator(tf);
    }

    public Iterator<Operand<T>> valIterator(Ops tf) {
        return this.val.batchIterator(tf);
    }

    public Iterator<Operand<T>> testIterator(Ops tf) {
        return this.test.batchIterator(tf);
    }

    private KerasDataset<T> makeTestSplit(double testSplit) {
        if (test != null) throw new IllegalStateException("Cannot override existing test split.");

        Pair<KerasSplit<T>, KerasSplit<T>> splitTrain = train.split(1 - testSplit);
        this.train = splitTrain.first();
        this.train = splitTrain.second();

        return this;
    }

    private KerasDataset<T> makeValSplit(double valSplit) {
        if (val != null) throw new IllegalStateException("Cannot override existing validation split.");

        Pair<KerasSplit<T>, KerasSplit<T>> splitTrain = train.split(1 - valSplit);
        this.train = splitTrain.first();
        this.val = splitTrain.second();

        return this;
    }

    public long trainSize() {
        return this.train.size();
    }

    public long testSize() {
        if (this.test == null) return 0;
        return this.test.size();
    }

    public long valSize() {
        if (this.test == null) return 0;
        return this.val.size();
    }
}
