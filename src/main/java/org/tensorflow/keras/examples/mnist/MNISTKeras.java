package org.tensorflow.keras.examples.mnist;

import org.tensorflow.Graph;
import org.tensorflow.data.TensorFrame;
import org.tensorflow.op.Ops;
import org.tensorflow.utils.Pair;

public class MNISTKeras {

  public static void main(String[] args) throws Exception {
    try (Graph graph = new Graph()) {
      Ops tf = Ops.create(graph);
      Pair<TensorFrame<Float>, TensorFrame<Float>>
    }
  }
}
