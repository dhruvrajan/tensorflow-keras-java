package io.gitlab.tensorflow.keras.mixin;

import org.tensorflow.Operand;

import java.util.ArrayList;
import java.util.List;

public interface Printable {
    List<Operand> printOps = new ArrayList<>();

    default List<Operand> getPrintOps() {
        return printOps;
    }

    default void addPrintOp(Operand printOp) {
        printOps.add(printOp);
    }
}
