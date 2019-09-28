package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.op.Ops;

public abstract class TensorFrame<T> implements Dataset<T>, GraphLoader<T> {
  protected long batchSize = 1;
  protected boolean dropRemainder = false;

  /* Override functions from Dataset<T> */
  @Override
  public Dataset<T> batch(long batchSize, boolean dropRemainder) {
    this.batchSize = batchSize;
    this.dropRemainder = dropRemainder;
    return this;
  }

  public long batchSize() {
    return this.batchSize;
  }

  public long numBatches() {
    return size() / batchSize;
  }
}
