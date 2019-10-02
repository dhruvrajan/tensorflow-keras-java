package org.tensorflow.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.keras.mixin.KerasType;
import org.tensorflow.keras.utils.Keras;
import org.tensorflow.keras.utils.TensorShape;
import org.tensorflow.op.Ops;

public class Flatten extends Layer<Float> implements KerasType<Float> {
    private static int FLATTEN_INPUT_LENGTH = 1;
    private int units;


    public Flatten(int units, Flatten.Options options) {
        super(FLATTEN_INPUT_LENGTH);
        this.units = units;
    }

    private Flatten() {
        super(FLATTEN_INPUT_LENGTH);
        this.units = units;
    }

    public static Flatten create() {
        return new Flatten();
    }

    public static Options options() {
        return new Flatten.Options();
    }

    public static class Options {
        // Default parameters
        public Options() {}
    }

    public void build(Ops tf, Shape inputShape) {
        this.built = true;
    }

    public Shape computeOutputShape(Shape inputShape) {
        // leaves unknown dimensions unknown
        return Shape.make(new TensorShape(inputShape).numElements());
    }

    @SafeVarargs
    public final Operand<Float> call(Ops tf, Operand<Float>... inputs) {
        return this.call(tf, inputs[0]);
    }

    private Operand<Float> call(Ops tf, Operand<Float> input) {
        return tf.reshape(input, tf.reduceProd(Keras.shapeOperand(tf, input.asOutput().shape()), tf.constant(0)));
    }
}
