// package io.gitlab.keras.utils;
//
// import org.nd4j.linalg.api.buffer.DataBuffer;
// import org.nd4j.linalg.api.buffer.FloatBuffer;
// import org.nd4j.linalg.api.ndarray.INDArray;
// import org.nd4j.linalg.factory.Nd4j;
// import org.nd4j.serde.binary.BinarySerde;
// import org.tensorflow.Tensor;
// import org.tensorflow.Tensors;
// public class ConversionUtils {
//
//    public static void main(String[] args) {
//
//        Tensor<Float> first = Tensors.create(new float[][] {{1, 3, 5},
//                {7, 9, 11}
//        });
//
//
//        DataBuffer dbf = new FloatBuffer(first.numElements());
//        first.writeTo(java.nio.FloatBuffer.allocate(first.numElements()));
//
//        //System.out.println(first.toString());
//        //System.out.println(TensorFromNDArray(NDArrayFromTensor(first)));
//    }
//
//    public static INDArray NDArrayFromTensor(Tensor tensor) {
//        DataBuffer dbf = new FloatBuffer(tensor.numElements());
//        tensor.writeTo(dbf.asNioFloat());
//        return Nd4j.create(dbf, tensor.shape());
//    }
//
//    public static Tensor<Float> TensorFromNDArray(INDArray ndArray) {
//        java.nio.FloatBuffer dbf = BinarySerde.toByteBuffer(ndArray).asFloatBuffer();
//        return Tensor.create(ndArray.shape(), dbf);
//    }
// }
