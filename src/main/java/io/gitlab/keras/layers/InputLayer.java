package io.gitlab.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class InputLayer extends Layer {
    public Placeholder<Float> iris;
    int length;
    int batchSize;
    public InputLayer(int length, int batchSize) {
        this.length = length;
        this.batchSize = batchSize;
    }

    public Operand build(Ops tf) {
        System.out.println("Input layer " + this.id + " length " + length);
        iris = tf.placeholder(Float.class, Placeholder.shape(Shape.make(batchSize, length)));
        return iris;
    }
}
