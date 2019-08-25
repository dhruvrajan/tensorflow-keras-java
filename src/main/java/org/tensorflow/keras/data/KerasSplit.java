package org.tensorflow.keras.data;

import org.tensorflow.Operand;
import org.tensorflow.utils.Pair;
import org.tensorflow.op.Ops;

import java.util.Iterator;

public abstract class KerasSplit<T> {
  public abstract Pair<KerasSplit<T>, KerasSplit<T>> split(double splitBy);

  public abstract Pair<KerasSplit<T>, KerasSplit<T>> split(int splitIndex);

  public abstract Iterator<Operand<T>> batchIterator(Ops tf, int batchSize);

  public abstract void shuffle();

  public abstract long size();
}
