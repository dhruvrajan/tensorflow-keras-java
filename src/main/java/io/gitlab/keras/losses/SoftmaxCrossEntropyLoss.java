package io.gitlab.keras.losses;

import org.tensorflow.Operand;
import org.tensorflow.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

import java.util.Collections;

import static io.gitlab.keras.utils.Keras.constArray;

public class SoftmaxCrossEntropyLoss extends Loss {
    public Operand<Float> loss;
    public Operand lossPrintOp;



    @Override
    protected Operand<Float> call(Ops tf, Operand<Float> actual, Placeholder<Float> labels) {
        loss = tf.math.mean(tf.nn.softmaxCrossEntropyWithLogits(actual, labels).loss(), tf.constant(0));
//        lossPrintOp = tf.print(loss, Collections.singletonList(loss));
        return loss;
    }

    @Override
    public void build(Ops tf, Shape inputShape) {

    }

    @Override
    public Shape computeOutputShape(Shape inputShape) {
        return Shape.unknown();
    }
}
