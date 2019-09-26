//package org.tensorflow.keras.examples.mnist;
//
//import org.tensorflow.Graph;
//import org.tensorflow.data.TensorFrame;
//import org.tensorflow.keras.datasets.MNISTLoader;
//import org.tensorflow.op.Ops;
//import org.tensorflow.utils.Pair;
//
//public class MNISTKeras {
//
//  public static void main(String[] args) throws Exception {
//    try (Graph graph = new Graph()) {
//      // Load MNIST Train / Test Data
//      Pair<TensorFrame<Float>, TensorFrame<Float>> data = MNISTLoader.loadData();
//      TensorFrame<Float> train = data.first();
//      TensorFrame<Float> test = data.second();
//
//      Ops tf = Ops.create(graph);
//    }
//  }
//}
