package org.tensorflow.data;

import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;

import java.util.Iterator;

public interface GraphLoader<T> extends BatchLoader<T> {
  /**
   * Get placeholder objects to feed dataset tensors into.
   * @return An array of Placeholder<T> objects representing the dataset in a tensorfow graph.
   */
  Placeholder<T>[] getDataPlaceholders();

  /**
   * Get the tensor objects associated with the tensor.
   *
   * @return An array of Tensor<T> objects matching this dataset's placeholders, with the dataset dataset information
   *         to be loaded into the tensorflow graph.
   */
  Tensor<T>[] getDataTensors();

  /**
   * Iterate over the batch operands from this dataset.
   *
   * @param tf Ops object greated from the current tensorflow graph.
   * @return an iterator over batch slices from this dataset.
   */
  Iterator<Operand<T>[]> getBatchOps(Ops tf);
}
