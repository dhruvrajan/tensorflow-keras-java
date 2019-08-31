package org.tensorflow.data;

import java.util.Iterator;

import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;

public interface EagerLoader<T> extends BatchLoader<T> {
    Iterator<Constant<T>> getBatchOps(Ops tf);
}
