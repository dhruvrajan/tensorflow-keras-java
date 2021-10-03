package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TType;

public abstract class TensorFrame<T extends TType> implements Dataset<T>, GraphLoader<T> {
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
