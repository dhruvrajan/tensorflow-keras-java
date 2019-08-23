package io.gitlab.tensorflow.keras.utils;

public class Pair<T, S> {
    private T first;
    private S second;

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
}
