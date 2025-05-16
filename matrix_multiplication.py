import numpy as np
from typing import Self

def next_pow2(n: int):
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()

class BaseMatrix:
    def __init__(self, M: np.ndarray):
        self.M = M

    def __add__(self, other: Self) -> Self:
        return type(self)(self.M + other.M)

    def __sub__(self, other: Self) -> Self:
        return type(self)(self.M - other.M)

    def block11(self) -> Self:
        """
        :return: The top-left block of the matrix.
        """
        dim0 = self.M.shape[0]//2
        dim1 = self.M.shape[1]//2
        matrix_block = self.M[:dim0, :dim1]
        return type(self)(matrix_block)

    def block12(self) -> Self:
        """
        :return: The top-right block of the matrix.
        """
        dim0 = self.M.shape[0]//2
        dim1 = self.M.shape[1]//2
        matrix_block = self.M[:dim0, dim1:]
        return type(self)(matrix_block)

    def block21(self) -> Self:
        """
        :return: The bottom-left block of the matrix.
        """
        dim0 = self.M.shape[0]//2
        dim1 = self.M.shape[1]//2
        matrix_block = self.M[dim0:, :dim1]
        return type(self)(matrix_block)

    def block22(self) -> Self:
        """
        :return: The bottom-right block of the matrix.
        """
        dim0 = self.M.shape[0]//2
        dim1 = self.M.shape[1]//2
        matrix_block = self.M[dim0:, dim1:]
        return type(self)(matrix_block)

    @classmethod
    def make_from_blocks(cls, A11: Self, A12: Self, A21: Self, A22: Self) -> Self:
        row1 = np.concatenate([A11.M, A12.M], axis=1)
        row2 = np.concatenate([A21.M, A22.M], axis=1)
        whole = np.concatenate([row1, row2], axis=0)
        return type(A11)(whole)

class NaiveMatrix(BaseMatrix):

    def __init__(self, M: np.ndarray):
        super().__init__(M)

    def __mul__(self, other: Self):

        # Stores the matrices
        A = self.M
        B = other.M

        # Prepares the new matrix
        C = np.zeros((A.shape[0], B.shape[1]))

        # Computes each element of the new array
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                row_dot_prod = 0
                for k in range(A.shape[1]):
                    row_dot_prod += A[i,k] * B[k,j]
                C[i,j] = row_dot_prod

        return NaiveMatrix(C)
    

class StrassenMatrix(BaseMatrix):

    def __init__(self, M: np.ndarray):
        super().__init__(M)

    def __mul__(self, other: Self):

        # Stores the matrices
        A = self.M
        B = other.M

        # Checks for the base case
        if A.size == 1 and B.size == 1:
            return StrassenMatrix(A @ B)

        # Computes the A matrix blocks
        A11 = self.block11()
        A12 = self.block12()
        A21 = self.block21()
        A22 = self.block22()

        # Computes the A matrix blocks
        B11 = other.block11()
        B12 = other.block12()
        B21 = other.block21()
        B22 = other.block22()

        # Computes the intermediate values
        P1 = (A11 + A22) * (B11 + B22)
        P2 = (A21 + A22) * (B11      )
        P3 = (      A11) * (B12 - B22)
        P4 = (      A22) * (B21 - B11)
        P5 = (A11 + A12) * (B22      )
        P6 = (A21 - A11) * (B11 + B12)
        P7 = (A12 - A22) * (B21 + B22)

        # Computes the C blocks
        C11 = P1 + P4 - P5 + P7
        C12 = P3 + P5
        C21 = P2 + P4
        C22 = P1 + P3 - P2 + P6

        return StrassenMatrix.make_from_blocks(C11, C12, C21, C22)


class WinogradMatrix(BaseMatrix):

    def __init__(self, M: np.ndarray):
        super().__init__(M)

    def __mul__(self, other: Self):

        # Stores the matrices
        A = self.M
        B = other.M

        # Checks for the base case
        if A.size == 1 and B.size == 1:
            return WinogradMatrix(A * B)

        # Computes the A matrix blocks
        A11 = self.block11()
        A12 = self.block12()
        A21 = self.block21()
        A22 = self.block22()

        # Computes the A matrix blocks
        B11 = other.block11()
        B12 = other.block12()
        B21 = other.block21()
        B22 = other.block22()

        # Computes the first set of intermediate values
        S1 = A21 + A22
        S2 = S1  - A11
        S3 = A11 - A21
        S4 = A12 - S2
        S5 = B12 - B11
        S6 = B22 - S5
        S7 = B22 - B12
        S8 = S6  - B21

        # Computes the second set of intermediate values
        M1 = S2  * S6
        M2 = A11 * B11
        M3 = A12 * B21
        M4 = S3  * S7
        M5 = S1  * S5
        M6 = S4  * B22
        M7 = A22 * S8

        # Computes the third set of intermediate values
        T1 = M1 + M2
        T2 = T1 + M4

        # Computes the C blocks
        C11 = M2 + M3
        C12 = T1 + M5 + M6
        C21 = T2 - M7
        C22 = T2 + M5

        return WinogradMatrix.make_from_blocks(C11, C12, C21, C22)


def test_products(tests: int, A_shape: tuple[int], B_shape: tuple[int]):
    
    # Runs the specified number of tests
    matrices = []
    ground_truths = []
    for test in range(tests):

        # Computes the dimension
        size = next_pow2(max(max(A_shape), max(B_shape)))

        # Prepares the first matrix for multiplication, with zero padding
        A = np.zeros((size, size))
        base_A = np.random.rand(*A_shape) * 10 - 5
        A[:A_shape[0], :A_shape[1]] = base_A

        # Prepares the second matrix for multiplication, with zero padding
        B = np.zeros((size, size))
        base_B = np.random.rand(*B_shape) * 10 - 5
        B[:B_shape[0], :B_shape[1]] = base_B
        
        # Computes the ground truth
        matrices.append((A,B))
        ground_truths.append(base_A @ base_B)

    # Computes the naive product
    print("Testing the Naive Matrix Multiplication Algorithm...")
    for (A,B), ground_truth in zip(matrices, ground_truths):
        naive_prod = (NaiveMatrix(A) * NaiveMatrix(B)).M[:A_shape[0], :B_shape[1]]
        assert np.all(np.abs(ground_truth - naive_prod) < 1e-4), "The naive matrix multiplication algorithm implementation is incorrect"

    # Computes the Strassen product
    print("Testing Strassen's Algorithm...")
    for (A,B), ground_truth in zip(matrices, ground_truths):
        strassen_prod = (StrassenMatrix(A) * StrassenMatrix(B)).M[:A_shape[0], :B_shape[1]]
        assert np.all(np.abs(ground_truth - strassen_prod) < 1e-4), "The Strassen's matrix multiplication algorith implementation is incorrect"

    # Computes the Winograd product
    print("Testing Winograd's Variant...")
    for (A,B), ground_truth in zip(matrices, ground_truths):
        winograd_prod = (WinogradMatrix(A) * WinogradMatrix(B)).M[:A_shape[0], :B_shape[1]]
        assert np.all(np.abs(ground_truth - winograd_prod) < 1e-4), "The Winograd's matrix multiplication algorith implementation is incorrect"

    print("All tests passed!")


if __name__ == "__main__":
    test_products(100, (12, 13), (13, 14))
    test_products(100, (16, 16), (16, 16))
    test_products(100, (12, 12), (12, 12))

    # All tests passed!