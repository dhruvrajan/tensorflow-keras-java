package io.gitlab.keras.losses;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;

import java.util.Collections;

import static io.gitlab.keras.utils.Keras.constArray;

public class SoftmaxCrossEntropyLoss extends Loss {
    public Operand<Float> loss;
    public Operand lossPrintOp;

    public Operand<Float> build(Ops tf, Operand<Float> actual, Operand<Float> labels) {
        loss = tf.mean(tf.softmaxCrossEntropyWithLogits(actual, labels).loss(), tf.constant(0));

        lossPrintOp = tf.print(loss, Collections.singletonList(loss));
        return loss;
    }


}
