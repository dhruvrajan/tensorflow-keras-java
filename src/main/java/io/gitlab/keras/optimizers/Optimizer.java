package io.gitlab.keras.optimizers;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;

import java.util.List;

public abstract class Optimizer<T> {
    List<Operand<Float>> targets;

    public abstract void build(Ops tf, List<Variable<T>> weights, Operand<T> loss);

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


