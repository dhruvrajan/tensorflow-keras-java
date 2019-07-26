package io.gitlab.keras.optimizers;

import org.tensorflow.Operand;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;

import java.util.ArrayList;
import java.util.List;

public class GradientDescentOptimizer extends Optimizer<Float> {
    Gradients gradients;
//    public List<Operand<Float>> targets;
    private final float LEARNING_RATE;

    public GradientDescentOptimizer(float lr) {
        targets = new ArrayList<>();
        LEARNING_RATE = lr;
    }

    public void build(Ops tf, List<Variable<Float>> weights, Operand<Float> loss) {
        tf = tf.withName("GradientDescentOptimizer");
        gradients = tf.gradients(loss, weights);
        Constant<Float> alpha = tf.constant(LEARNING_RATE);

        for (int i = 0; i < weights.size(); i++) {
            targets.add(tf.applyGradientDescent(weights.get(i), alpha, gradients.dy(i)));
//            targets.add(tf.applyAdam(weights.get(i),
//                    constArray(tf, 0),
//                    constArray(tf, 0),
//                    constArray(tf, 1),
//                    constArray(tf, 1),
//                    constArray(tf, 0.001),
//                    constArray(tf, 0.9),
//                    constArray(tf, 0.99), constArray(tf, 1e-8),gradients.dy(i)));//(weights.get(i), alpha, gradients.dy(i)));

        }
    }


}
