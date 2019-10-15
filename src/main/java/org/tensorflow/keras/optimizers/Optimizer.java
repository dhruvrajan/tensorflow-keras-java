package org.tensorflow.keras.optimizers;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Variable;

import java.util.List;

public abstract class Optimizer<T> {
  protected List<Operand<T>> targets;

  public List<Operand<T>> minimize(Ops tf, Operand<T> loss, List<Variable<T>> weights) {
    Gradients gradients = computeGradients(tf, loss, weights);
    return applyGradients(tf, weights, gradients);
  }

  public Gradients computeGradients(Ops tf, Operand<T> loss, List<Variable<T>> weights) {
    return tf.gradients(loss, weights);
  }

  public abstract  void build(Ops tf);

  public abstract List<Operand<T>> applyGradients(
      Ops tf, List<Variable<T>> weights, Gradients gradients);

  public List<Operand<T>> getTargets() {
    return this.targets;
  }

  public List<Operand<T>> trainingOps() {
    return targets;
  }
}
