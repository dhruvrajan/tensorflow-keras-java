package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;

import java.util.Iterator;

public interface GraphLoader<T> extends Dataset<T>, AutoCloseable {
  void build(Ops tf);
  Operand<T>[] getBatchOperands();
  void addBatchToSessionRunner(Ops tf, Session.Runner runner, long batchIndex, boolean fetchBatches);
}
