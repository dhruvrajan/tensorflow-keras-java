package org.tensorflow.keras.optimizers;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Variable;

import java.util.ArrayList;
import java.util.List;

public class GradientDescentOptimizer<T extends Number> extends Optimizer<T> {
    private final float lr;
    private Constant<T> alpha;

    public GradientDescentOptimizer(float lr) {
        this.lr = lr;
    }

    @Override
    public void build(Ops tf) {
        this.alpha = tf.constant(lr, getDtype());
    }

    @Override
    public List<Operand<T>> applyGradients(Ops tf, List<Variable<T>> weights, Gradients gradients) {
        List<Operand<T>> targets = new ArrayList<>();
        for (int i = 0; i < weights.size(); i++) {
            targets.add(tf.applyGradientDescent(weights.get(i), alpha, gradients.dy(i)));
        }

        return targets;
    }
}


