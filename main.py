import pygame as p
import math
import numpy as np

p.init()

# Display Settings
display_width = 800
display_height = 600
display = p.display.set_mode((display_width, display_height))
CAPTION = "3d"
p.display.set_caption(CAPTION)

# Matrix functions
def Dot(matrix1, matrix2):
    matrix1_rows, matrix1_columns = np.shape(matrix1)
    matrix2_rows, matrix2_columns = np.shape(matrix2)
    if matrix1_columns != matrix2_rows:
        raise ValueError("You cannot dot product matrices where the first matrix column is of different dimension to the second matrix row")
    new_matrix = np.zeros((matrix1_rows, matrix2_columns))  # Creates an empty new_matrix which we will store the result in
    return new_matrix
#Testing
a = np.array([
    [1, 2],
    [2, 4],
    [4, 5]
])
b = np.array([
    [1, 4, 3],
    [2, 5, 4],
])
print(Dot(a, b))
