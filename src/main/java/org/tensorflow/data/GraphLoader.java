package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.utils.Pair;

import java.util.Iterator;

public interface GraphLoader<T> extends BatchLoader<T> {
  Placeholder<T>[] getPlaceholders();

  Iterator<Pair<Tensor<T>[], Operand<T>[]>> getBatchTensorsAndOps(Ops tf);
}
