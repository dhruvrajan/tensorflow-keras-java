package org.tensorflow.keras.layers;

import org.tensorflow.Shape;
import org.tensorflow.Tensor;

import java.util.List;
import java.util.Map;

public class InputSpec {
  Class dtype;
  Shape shape;
  int ndim;
  int maxNdim;
  int minNdim;

  Map<Integer, Integer> axes;

  public InputSpec InputSpec() {
    return this;
  }

  public InputSpec dtype(Class dtype) {
    this.dtype = dtype;
    return this;
  }

  public InputSpec shape(Shape shape) {
    this.shape = shape;
    return this;
  }

  public InputSpec ndim(int ndim) {
    this.ndim = ndim;
    return this;
  }

  public InputSpec maxNdim(int maxNdim) {
    this.maxNdim = maxNdim;
    return this;
  }

  public InputSpec minNdim(int minNdim) {
    this.minNdim = minNdim;
    return this;
  }

  public InputSpec axes(Map<Integer, Integer> axes) {
    this.axes = axes;
    return this;
  }

  public static void assertInputCompatibility(
      InputSpec inputSpec, List<Tensor> inputs, String layerName) {
    throw new UnsupportedOperationException("Not yet implemented");
  }
}
