// package io.gitlab.keras.metrics;
//
// import org.tensorflow.Operand;
// import org.tensorflow.Shape;
// import org.tensorflow.op.Ops;
// import org.tensorflow.op.core.Placeholder;
//
// public class BinaryAccuracy extends Metric {
//    private static final float DEFAULT_THRESHOLD = 0.5f;
//    private float threshold = DEFAULT_THRESHOLD;
//
//    BinaryAccuracy setThreshold(float threshold) {
//        this.threshold = threshold;
//        return this;
//    }
//
//    @Override
//    public Operand<Float> call(Ops tf, Operand<Float> output, Placeholder<Float> label) {
//        Operand<Float> yPred = tf.dtypes.cast(tf.math.greater(output,
// tf.constant(threshold)).asOutput(), Float.class);
//        Operand<Float> equal = tf.dtypes.cast(tf.math.equal(label, yPred).asOutput(),
// Float.class);
//        return tf.math.mean(equal, tf.constant(-1));
//
//    }
//
//    @Override
//    public void create(Ops tf, Shape inputShape) {
//
//    }
//
//    @Override
//    public Shape computeOutputShape(Shape inputShape) {
//        return Shape.unknown();
//    }
// }
