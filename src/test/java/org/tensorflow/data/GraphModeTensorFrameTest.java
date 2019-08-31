package org.tensorflow.data;

import org.junit.jupiter.api.Test;
import org.tensorflow.*;
import org.tensorflow.op.Ops;
import org.tensorflow.utils.Pair;
import org.tensorflow.utils.SessionRunner;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class GraphModeTensorFrameTest {
    @Test
    void testGraphModeSession() {
        Tensor<Float> XTensor = Tensors.create(
                new float[][] {
                        {1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9},
                        {10, 11, 12},
                        {13, 14, 15},
                        {16, 17, 18},
                        {19, 20, 21},
                        {22, 23, 24},
                        {25, 26, 27}
                }
        );
        Tensor<Float> yTensor = Tensors.create(
                new float[] {0, 1, 2, 3, 4, 5, 6, 7, 8}
        );


        GraphModeTensorFrame<Float> graphModeTensorFrame
                = new GraphModeTensorFrame<>(Float.class, XTensor, yTensor);

        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);
            for (Pair<Tensor<Float>[], Operand<Float>[]> batch :
                    (Iterable<Pair<Tensor<Float>[], Operand<Float>[]>>) () -> graphModeTensorFrame.getBatchTensorsAndOps(tf)) {

                Tensor<Float>[] tensors = batch.first();
                Operand<Float>[] ops = batch.second();

                SessionRunner runner = new SessionRunner(new Session(graph).runner());

                List<Tensor<?>> outputs = runner
                        .feed(tensors, ops)
                        .fetch(ops)
                        .run();

                Tensor<?> XBatch = outputs.get(0);
                Tensor<?> yBatch = outputs.get(1);

                FloatBuffer XBuffer = FloatBuffer.allocate(3 * 2);
                FloatBuffer yBuffer = FloatBuffer.allocate(2);

                XBatch.writeTo(XBuffer);
                yBatch.writeTo(yBuffer);

                System.out.println("XBATCH: " + Arrays.toString(XBuffer.array()));
                System.out.println("YBATCH: " + Arrays.toString(yBuffer.array()));
            }
        }

    }
}