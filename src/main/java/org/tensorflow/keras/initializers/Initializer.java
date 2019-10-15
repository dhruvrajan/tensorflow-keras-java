package org.tensorflow.keras.initializers;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;

public abstract class Initializer<T> {
  protected Assign<T> initializerOp;

  protected boolean built = false;

  public abstract Operand<T> build(Ops tf, Operand<T> in, Class<T> dtype);

  public Assign<T> getInitializerOp() {
    return initializerOp;
  }

  public boolean isBuilt() {
    return this.built;
  }
}
