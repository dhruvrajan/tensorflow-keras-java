package io.gitlab.keras.examples.eager;

import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.op.Ops;

import java.nio.FloatBuffer;
import java.util.Arrays;

public class EagerPractice {

    public static void main(String[] args) {




        try (EagerSession session = EagerSession.create()) {

            Ops tf = Ops.create(session);

            var t1 = tf.constant(new float[][]{
                    {1, 2, 3},
                    {4, 5, 6}
            });

            var t2 = tf.constant(new float[][] {
                    {2, 4},
                    {6, 8},
                    {10, 12}
            });

            Operand<Float> out = tf.linalg.matMul(t1, t2);
            System.out.println(out.asOutput().toString());
            printTensor(out.asOutput().tensor());

        }



    }

    private static void printTensor(Tensor<Float> floatTensor) {
        long[] shape = floatTensor.shape();

        int prod = 1;
        for (int i = 0; i < shape.length; i++) {
            prod *= shape[i];
        }

        var arr = FloatBuffer.allocate(prod);
        floatTensor.writeTo(arr);
        System.out.println(Arrays.toString(arr.array()));
        var floatarr = new float[(int) shape[0]][(int) shape[1]];
        for (int i = 0; i < floatarr.length; i++) {
            arr.get(floatarr[i]);
        }

        for (int i = 0; i < floatarr.length; i++) {
            System.out.println(Arrays.toString(floatarr[i]));
        }

        }
}

