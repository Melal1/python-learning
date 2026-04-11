import numpy as np


def main():
    a = np.eye(4, dtype=int)
    print(a)

    print()

    b = np.eye(3, 4, dtype=int)
    print(b)

    print()

    c = np.diag(b + b)

    print(c)

    print()

    d = np.ones(4, dtype=int)

    print(d)

    print()

    e = np.zeros(4, dtype=int)

    print(e)

    a1 = np.array([1, 2, 3])
    print()
    print(type(a))
    print()
    print(a1.shape)
    print()
    print(a1[0], a1[1], a1[2])
    a1[0] = 5
    print()
    print(a1)
    b1 = np.array([[1, 2, 3], [4, 5, 6]])
    print()
    print(b1.shape)
    print()
    print(b1[0, 0], b1[0, 1], b1[1, 0])
    print()
    print(b1)

    # --- 1. SETUP DATA ---
    # Matrices (2D arrays)
    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([[5, 6], [7, 8]], dtype=np.float64)

    # Vectors (1D arrays)
    v = np.array([9, 10])
    w = np.array([11, 12])

    # --- 2. ELEMENT-WISE OPERATIONS ---
    # These operations apply the math to each individual position (e.g., top-left + top-left)

    # Addition: [[1+5, 2+6], [3+7, 4+8]] -> [[6, 8], [10, 12]]
    print("Addition:")
    print(x + y)
    print(np.add(x, y))

    # Subtraction: [[1-5, 2-6], [3-7, 4-8]] -> [[-4, -4], [-4, -4]]
    print("\nSubtraction:")
    print(x - y)
    print(np.subtract(x, y))

    # Element-wise Multiplication: [[1*5, 2*6], [3*7, 4*8]] -> [[5, 12], [21, 32]]
    # Note: This is NOT matrix multiplication!
    print("\nElement-wise Multiply:")
    print(x * y)
    print(np.multiply(x, y))

    # --- 3. DOT PRODUCTS & MATRIX MATH ---
    # This follows the rules of Linear Algebra (Row × Column)

    # Vector Dot Product: (9*11) + (10*12) = 99 + 120 = 219
    print("\nVector Dot Product:")
    print(v.dot(w))
    print(np.dot(v, w))

    # Matrix Dot Product (Matrix Multiplication):
    # Top-left result: (Row 1 of x) dot (Col 1 of y) -> (1*5 + 2*7) = 19
    print("\nMatrix Dot Product:")
    print(x.dot(y))


if __name__ == "__main__":
    main()
