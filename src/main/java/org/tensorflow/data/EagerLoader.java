package org.tensorflow.data;

import org.tensorflow.op.core.Constant;

public abstract class EagerLoader<T> {
    public abstract Constant<T> getBachOps(long i);
}
