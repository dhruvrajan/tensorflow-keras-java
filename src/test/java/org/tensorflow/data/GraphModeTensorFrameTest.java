package org.tensorflow.data;

import org.junit.jupiter.api.Test;
import org.tensorflow.*;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.utils.Pair;
import org.tensorflow.utils.SessionRunner;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.List;

class GraphModeTensorFrameTest {
    @Test
    void testGraphModeSession() {
        int batchSize = 2;
        try (Graph graph = new Graph();
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
                     new float[] {0, 1, 2, 3, 4, 5, 6, 7, 8})) {


            GraphModeTensorFrame<Float> graphModeTensorFrame
                    = new GraphModeTensorFrame<>(Float.class, XTensor, yTensor);


            Ops tf = Ops.create(graph);

            graphModeTensorFrame.build(tf);
            graphModeTensorFrame.batch(batchSize);
            Placeholder<Float>[] placeholders = graphModeTensorFrame.getPlaceholders();

            int batches = 0;
            for (Pair<Tensor<Float>[], Operand<Float>[]> batch : (Iterable<Pair<Tensor<Float>[], Operand<Float>[]>>) () -> graphModeTensorFrame.getBatchTensorsAndOps(tf)) {

                try (Session session = new Session(graph)) {
                    SessionRunner runner = new SessionRunner(session.runner());
                    Tensor<Float>[] tensors = batch.first();
                    Operand<Float>[] ops = batch.second();
                    List<Tensor<?>> outputs = runner.feed(tensors, placeholders).fetch(ops).run();

                    try (Tensor<?> XBatch = outputs.get(0);
                         Tensor<?> yBatch = outputs.get(1); ) {

                        FloatBuffer XBuffer = FloatBuffer.allocate(3 * 2);
                        FloatBuffer yBuffer = FloatBuffer.allocate(2);

                        XBatch.writeTo(XBuffer);
                        yBatch.writeTo(yBuffer);

                        float[] xarray = XBuffer.array();

                        for (int i = 0; i < XBatch.shape()[0]; i++) {
                            for (int j = 0; j < XBatch.shape()[1]; j++) {
                                float real = xarray[i * batchSize + j];
                                assertEquals(batches * batchSize * XBatch.shape()[1] + i * batchSize + j + 1, real);
                            }
                        }
                    }

                    batches ++;
                }
            }
        }
    }
}