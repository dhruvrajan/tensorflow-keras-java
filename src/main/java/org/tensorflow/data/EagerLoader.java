package org.tensorflow.data;

import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;

import java.util.Iterator;

public interface EagerLoader<T> extends BatchLoader<T> {
  Iterator<Constant<T>> getBatchOps(Ops tf);
}
