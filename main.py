#import pygame as p
import math
import numpy as np

#p.init()

# Display Settings
display_width = 800
display_height = 600
# display = p.display.set_mode((display_width, display_height))
CAPTION = "3d"
# p.display.set_caption(CAPTION)

# Matrix functions
def Dot(matrix1, matrix2):  # Calculates the dot product of two matrices
    matrix1_rows, matrix1_columns = np.shape(matrix1)
    matrix2_rows, matrix2_columns = np.shape(matrix2)
    if matrix1_columns != matrix2_rows:
        raise ValueError("You cannot dot product matrices where the first matrix column is of different dimension to the second matrix row")
    new_matrix = np.zeros((matrix1_rows, matrix2_columns))  # Creates an empty new_matrix which we will store the result in
    for i in range(matrix1_rows):
        for j in range(matrix2_columns):
            new_matrix[i, j] = np.sum(matrix1[i]*matrix2[:, j])
    return new_matrix

def Cross(matrix1, matrix2):  # Calculates the cross product of two (3, 1) matrices
    ax, ay, az = matrix1
    bx, by, bz = matrix2
    cx, cy, cz = ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx
    return np.array([cx, cy, cz])

# Camera
camera_position = np.array([0, 0, 0])  # Initially at the origin, position given as a coordinate
camera_direction = np.array([1, 0, 0])  # Initially looking parallel to the positive x-axis, direction is given as a unit vector

#Objects
class Triangle:
    def __init__(self, v1, v2, v3):  # Vertices of the triangle, (should be np.array of shape (3, 1))
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    @property
    def normal(self):
        return np.cross(self.v2-self.v1, self.v3-self.v1)

#Rendering
for triangle in triangles:
    vertices = [triangle.v1, triangle.v2, triangle.v3]
    vectors_to_camera = []
    for vertex in vertices:
        vectors_to_camera.append(camera_position - vertex)

for vector in vectors_to_camera:
    
#Testing
triangles = []
tri1 = Triangle(np.array([1, 2, 3]), np.array([0, 0, 1]), np.array([4, 1, 2]))
triangles.append(tri1)
