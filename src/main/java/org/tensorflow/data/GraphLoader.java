package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.Tensor;


public interface GraphLoader<T> {
    public abstract Tensor<T>[] getBatchTensors(long i);
    public abstract Operand<T>[] getBatchOps();
}
