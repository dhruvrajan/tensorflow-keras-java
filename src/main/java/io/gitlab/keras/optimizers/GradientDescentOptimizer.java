package io.gitlab.keras.optimizers;

import org.tensorflow.Operand;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;

import java.util.ArrayList;
import java.util.List;

public class GradientDescentOptimizer extends Optimizer<Float> {
    private final float LEARNING_RATE;

    public GradientDescentOptimizer(float lr) {
        this.targets = new ArrayList<>();
        this.LEARNING_RATE = lr;
    }

    public List<Operand<Float>> applyGradients(Ops tf, List<Variable<Float>> weights, Gradients gradients) {
        tf = tf.withName("GradientDescentOptimizer");

        Constant<Float> alpha = tf.constant(LEARNING_RATE);
        for (int i = 0; i < weights.size(); i++) {
            targets.add(tf.train.applyGradientDescent(weights.get(i), alpha, gradients.dy(i)));
        }

        return targets;
    }


}
