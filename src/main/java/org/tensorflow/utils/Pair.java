package org.tensorflow.utils;

import java.util.Iterator;

public class Pair<T, S> {
  private final T first;
  private final S second;

  public Pair(T t, S s) {
    this.first = t;
    this.second = s;
  }

  public T first() {
    return first;
  }

  public S second() {
    return second;
  }

  public static <T, S> Pair<T, S> of(T t, S s) {
    return new Pair<>(t, s);
  }

  public static <T, S> Iterator<Pair<T, S>> zip(T[] first, S[] second) {
    int length = Math.min(first.length, second.length);
    return new Iterator<>() {
      int index = 0;

      @Override
      public boolean hasNext() {
        return index < length;
      }

      @Override
      public Pair<T, S> next() {
        index++;
        return Pair.of(first[index - 1], second[index - 1]);
      }
    };
  }
}
