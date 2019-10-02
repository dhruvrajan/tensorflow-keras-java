package org.tensorflow.keras.optimizers;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Variable;

import java.util.ArrayList;
import java.util.List;

public class GradientDescentOptimizer extends Optimizer<Float> {
    private final float LEARNING_RATE;
    Constant<Float> alpha;

    public GradientDescentOptimizer(float lr) {
        this.targets = new ArrayList<>();
        this.LEARNING_RATE = lr;
    }

    @Override
    public void build(Ops tf) {
        this.alpha = tf.constant(LEARNING_RATE);
    }

    public List<Operand<Float>> applyGradients(
            Ops tf, List<Variable<Float>> weights, Gradients gradients) {
        tf = tf.withName("GradientDescentOptimizer");

        for (int i = 0; i < weights.size(); i++) {
            targets.add(tf.applyGradientDescent(weights.get(i), alpha, gradients.dy(i)));
        }

        return targets;
    }
}
