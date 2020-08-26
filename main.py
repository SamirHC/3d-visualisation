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
    if np.shape(matrix1)[0] != np.shape(matrix2)[1]:
        raise ValueError("You cannot dot product matrices where the first matrix row is of different dimension to the second matrix column")
    
#Testing
a = np.array([
    [1, 2],
    [2, 4],
    #[4, 5]
])
b = np.array([
    [1, 4, 3],
    [2, 5, 4],
])

Dot(a, b)
