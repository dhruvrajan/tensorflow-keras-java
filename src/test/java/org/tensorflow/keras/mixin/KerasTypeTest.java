package org.tensorflow.keras.mixin;

import org.junit.jupiter.api.Test;
import org.tensorflow.keras.layers.Dense;

class KerasTypeTest {

  @Test
  void build() {}

  @Test
  void computeOutputShape() {}

  @Test
  void call() {
    var dense = Dense.create(4);
  }
}
