package io.gitlab.keras.optimizers;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Variable;

import java.util.ArrayList;
import java.util.List;

public class AdamOptimizer {
    // default parameters
    float LEARNING_RATE = 0.001f;
    float BETA1 = 0.9f;
    float BETA2 = 0.999f;

    static float DEFAULT_LEARNING_RATE = 0.001f;
    static float DEFAULT_BETA1 = 0.9f;
    static float DEFAULT_BETA2 = 0.999f;
    static float DEFAULT_EPSILON = 0;
    static boolean DEFAULT_AMSGRAD = false;

    Gradients gradients;
    public List<Operand<Float>> targets;
    private float learningRate;

    public AdamOptimizer(float lr) {
        targets = new ArrayList<>();
        learningRate = lr;
    }

    public void build(Ops tf, List<Variable<Float>> weights, Operand<Float> loss) {
        tf = tf.withName("GradientDescentOptimizer");
        gradients = tf.gradients(loss, weights);
        Constant<Float> alpha = tf.constant(learningRate);

    }
    private static Operand<Double> constArray(Ops tf, double... i) { return tf.constant(i); }

    static AdamOptimizer createDefault() {
        return new AdamOptimizer(DEFAULT_LEARNING_RATE);
    }
}
