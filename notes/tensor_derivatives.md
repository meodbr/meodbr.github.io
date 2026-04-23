# Tensor Derivatives: Shape Intuition Cheat Sheet

## Core Principle

Let:
- Input tensor has shape **S**
- Output tensor has shape **T**

Then:
\[
\frac{\partial (\text{output})}{\partial (\text{input})}
\text{ has shape } T + S
\]
("+" means concatenation of shapes)

---

## Index-Based Rule (Most Reliable)

Write derivatives with indices:
\[
\frac{\partial y_i}{\partial A_{j,k}}
\]

The resulting tensor carries all indices:
- Output indices first
- Input indices after

This directly determines the shape.

---

## Common Cases

### 1. Vector → Vector
\[
f: \mathbb{R}^n \to \mathbb{R}^m
\quad \Rightarrow \quad
\frac{\partial f}{\partial x} \in \mathbb{R}^{m \times n}
\]

(Standard Jacobian)

---

### 2. Matrix → Vector
\[
A \in \mathbb{R}^{m \times n}, \quad y = Ax \in \mathbb{R}^m
\]

\[
\frac{\partial y}{\partial A} \in \mathbb{R}^{m \times m \times n}
\]

(Indices: \(\partial y_i / \partial A_{j,k}\))

---

### 3. Matrix → Scalar
\[
L(A) \in \mathbb{R}
\quad \Rightarrow \quad
\frac{\partial L}{\partial A} \in \mathbb{R}^{m \times n}
\]

(This is the gradient; same shape as \(A\))

---

## Practical Rule (Most Used in ML)

If the output is a scalar \(L\), then:
\[
\nabla_A L \text{ has the same shape as } A
\]

---

## Why Higher-Order Tensors Rarely Appear

Full derivatives often produce higher-order tensors (e.g. 3D or more), but:

- In practice, we apply the **chain rule**
- This contracts dimensions and reduces back to the input shape

Example:
\[
L = g(Ax)
\quad \Rightarrow \quad
\frac{\partial L}{\partial A} = \left(\frac{\partial L}{\partial y}\right) x^\top
\]

---

## Key Intuition

Each derivative entry answers:
\[
\frac{\partial (\text{one output component})}{\partial (\text{one input component})}
\]

The full derivative is just all such entries arranged into a tensor.

---

## Reliable Strategy

When unsure:

1. Write a single entry:
   \[
   \frac{\partial (\cdot)}{\partial (\text{one element})}
   \]

2. Track remaining indices

3. Reconstruct the full shape

This method is always correct.