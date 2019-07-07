package io.gitlab.keras.layers;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

public class InputLayer extends Layer<Float> {
    public Placeholder<Float> input;
    private int length;
    private int batchSize;

    public InputLayer(int length, int batchSize) {
        this.length = length;
        this.batchSize = batchSize;

    }

    @Override
    public void build(Ops tf) {
        input = tf.placeholder(Float.class, Placeholder.shape(Shape.make(batchSize, length)));
    }

    @Override
    public Operand<Float> call(Ops tf, Operand<Float> in) {
        return input;
    }
}
