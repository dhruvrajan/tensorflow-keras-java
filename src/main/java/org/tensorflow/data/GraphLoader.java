package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TType;

public interface GraphLoader<T extends TType> extends Dataset<T> {

  Operand<T>[] getBatchOperands();

  void build(Ops tf);

  long size();

  Session.Runner feedSessionRunner(Session.Runner runner, long batch);
}
