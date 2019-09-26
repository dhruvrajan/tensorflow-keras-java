package org.tensorflow.keras.optimizers;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Variable;

import java.util.List;

public abstract class Optimizer<T> {
  protected List<Operand<Float>> targets;

  public List<Operand<Float>> minimize(Ops tf, Operand<Float> loss, List<Variable<Float>> weights) {
    Gradients gradients = computeGradients(tf, loss, weights);
    return applyGradients(tf, weights, gradients);
  }

  public Gradients computeGradients(Ops tf, Operand<Float> loss, List<Variable<Float>> weights) {
    return tf.gradients(loss, weights);
  }

  public abstract  void build(Ops tf);

  public abstract List<Operand<Float>> applyGradients(
      Ops tf, List<Variable<Float>> weights, Gradients gradients);

  public List<Operand<Float>> getTargets() {
    return this.targets;
  }

  public static Optimizer<Float> select(String optimizerType) {
    return Optimizers.select(Optimizers.valueOf(optimizerType));
  }

  public List<Operand<Float>> trainingOps() {
    return targets;
  }
}
