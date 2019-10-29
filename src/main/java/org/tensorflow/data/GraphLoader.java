package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.utils.SessionRunner;

import java.util.Iterator;

public interface GraphLoader<T> extends Dataset<T> {
  Iterator<Tensor<?>[]> batchIterator(Ops tf);
}
