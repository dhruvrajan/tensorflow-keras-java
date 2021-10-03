package org.tensorflow.utils;

import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.types.family.TType;

import java.util.List;

public class SessionRunner {
  private Session.Runner runner;

  public SessionRunner(Session.Runner runner) {
    this.runner = runner;
  }

  public SessionRunner addTarget(String operation) {
    this.runner.addTarget(operation);
    return this;
  }

  public SessionRunner addTarget(Operand operand) {
    this.runner.addTarget(operand);
    return this;
  }

  public SessionRunner addTargets(Operand<?>... operands) {
    for (Operand<?> operand : operands) {
      this.runner.addTarget(operand);
    }
    return this;
  }

  public <T extends TType> SessionRunner addTargets(List<Operand<T>> operands) {
    for (Operand<?> operand : operands) {
      this.runner.addTarget(operand);
    }
    return this;
  }

  public SessionRunner addTargets(String... targets) {
    for (String target : targets) {
      this.runner.addTarget(target);
    }

    return this;
  }

  public SessionRunner feed(Tensor[] tensors, Operand[] ops) {
    for (Pair<Tensor, Operand> pairs :
        (Iterable<Pair<Tensor, Operand>>) () -> Pair.zip(tensors, ops)) {
      this.runner.feed(pairs.second().asOutput(), pairs.first());
    }
    return this;
  }

  public SessionRunner fetch(String operation) {
    this.runner.fetch(operation);
    return this;
  }

  public SessionRunner fetch(Operand operand) {
    this.runner.fetch(operand);
    return this;
  }

  public SessionRunner fetch(String... operations) {
    for (String operation : operations) {
      this.runner.fetch(operation);
    }
    return this;
  }

  public SessionRunner fetch(Operand<?>... operands) {
    for (Operand<?> operand : operands) {
      this.runner.fetch(operand);
    }
    return this;
  }

  public <T extends TType> SessionRunner fetch(List<Operand<T>> operands) {
    for (Operand<?> operand : operands) {
      this.runner.fetch(operand);
    }
    return this;
  }

  public List<Tensor/*<?>*/> run() {
    return this.runner.run();
  }
}
