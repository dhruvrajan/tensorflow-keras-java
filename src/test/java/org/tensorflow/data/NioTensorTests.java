// package org.tensorflow.data;
//
// import org.junit.jupiter.api.Test;
// import org.tensorflow.nio.nd.FloatNdArray;
// import org.tensorflow.nio.nd.NdArrays;
// import org.tensorflow.nio.nd.Shape;
//
// import java.nio.FloatBuffer;
// import java.util.Arrays;
//
// public class NioTensorTests {
//
//    @Test
//    void test() {
//        float[] values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//        FloatNdArray matrix = NdArrays.ofFloats(Shape.create(2, 5));
//        FloatNdArray morebatches = NdArrays.ofFloats(Shape.create(5, 2));
//
//        matrix.write(values);
//
//
//
//        float[] out = new float[5];
//        System.out.println(matrix.at(1).get(0));
//        System.out.println("hey: " + Arrays.toString(out));
//    }
// }
