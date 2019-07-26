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
        return OptimizerType.select(OptimizerType.valueOf(optimizerType));
    }

    enum OptimizerType {
        sgd,
        adam,
        adagrad,
        adadelta;

        static Optimizer<Float> select(OptimizerType optimizerType) {
            switch (optimizerType) {
                case sgd:
                    return new GradientDescentOptimizer(0.2f);
                default:
                    throw new IllegalArgumentException("Invalid Optimizer Type.");
            }
        }
    }

}


