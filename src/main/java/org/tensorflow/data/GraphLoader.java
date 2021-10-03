package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.types.family.TType;
import org.tensorflow.utils.SessionRunner;

import java.util.Iterator;

public interface GraphLoader<T extends TType> extends Dataset<T> {

  Operand<T>[] getBatchOperands();

  void build(Ops tf);

  long size();

  Session.Runner  feedSessionRunner(Session.Runner runner, long batch);
}
