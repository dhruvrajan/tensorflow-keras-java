package io.gitlab.keras.utils;

import org.tensorflow.*;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;

import java.util.Arrays;
import java.util.Iterator;

public class TensorDataset<T> {
    private TensorSplit<T> train;
    private TensorSplit<T> test;

    public TensorDataset(TensorSplit<T> train, TensorSplit<T> test) {
        this.train = train;
        this.test = test;
    }
}
