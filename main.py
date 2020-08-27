import pygame as p
import math
import numpy as np
import time

p.init()

# Display Settings
display_width = 800
display_height = 600
display = p.display.set_mode((display_width, display_height))
CAPTION = "3d"
p.display.set_caption(CAPTION)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Initial Camera Settings
alpha = 0  # Angle in the xy plane
beta = 0 # Angle in the xz plane
gamma = 0  # Angle in the yz plane
x0 = 0  # x coordinate of camera
y0 = 0  # y coordinate of camera
z0 = 0  # z coordinate of camera

# Vector Calculations
def unitVector(vector):
    return vector / np.linalg.norm(vector)

def angleBetweenVectors(v1, v2):  # Returns angle between vectors measured in radians
    v1_u = unitVector(v1)
    v2_u = unitVector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def intersectionOfLineAndPlane(lineData, planeData):  # Returns the point (if any) of intersection between a line and a plane
    # lineData is in the form :: [p, v] where p is the point and v is a vector, given as np.arrays
    # planeData is in the form :: [p, n] as above, but n is the normal vector to the plane
    V0 = planeData[0]
    P0 = lineData[0]
    n = planeData[1]
    u = lineData[1]
    if np.dot(n, u) == 0:
        return np.array([nan, nan, nan])
    w = P0 - V0
    s = -np.dot(n, w) / np.dot(n, u)  # Calculates the value of the parametric variable
    return P0 + s*u
    
# Camera
def getAxisMatrix(ALPHA, BETA, GAMMA):
    alpha_rotation_matrix = np.array([
        [math.cos(ALPHA), -math.sin(ALPHA), 0],
        [math.sin(ALPHA), math.cos(ALPHA), 0],
        [0, 0, 1]
        ])  # Matrix required to rotate a plane by ALPHA in the xy plane
    beta_rotation_matrix = np.array([
        [math.cos(BETA), 0, -math.sin(BETA)],
        [0, 1, 0],
        [math.sin(BETA), 0, math.cos(BETA)]
        ])  # Matrix required to rotate a plane by ALPHA in the xz plane
    gamma-rotation_matric = np.array([
        [1, 0, 0],
        [0, math.cos(GAMMA), -math.sin(GAMMA)],
        [0, math.sin(GAMMA), math.cos(GAMMA)]
        ])  # Matrix required to rotate a plane by ALPHA in the yz plane
    axisMatrix = np.dot(np.dot(alpha_rotation_matrix, beta_rotation_matrix), gamma_rotation_matrix)
        
def findCameraDirection(ALPHA, BETA, GAMMA):
    i = math.sin(ALPHA)*math.sin(GAMMA) - math.cos(ALPHA)*math.sin(BETA)*math.cos(GAMMA)
    j = -math.cos(ALPHA)*math.sin(GAMMA) - math.sin(ALPHA)*math.sin(BETA)*math.cos(GAMMA)
    k = math.cos(BETA)*math.cos(GAMMA)
    return np.array([i, j, k])

def getInverseAxisMatrix(ALPHA, BETA, GAMMA):
    inverseMatrix = np.linalg.inv(getAxisMatrix(ALPHA, BETA, GAMMA))
    return inverseMatrix
        
camera_position = np.array([x0, y0, z0])  # Initially at the origin, position given as a coordinate np.array
camera_direction = findCameraDirection(alpha, beta, gamma)  # Initially looking parallel to the positive x-axis, direction is given as a unit vector np.array
camera_screen = [camera_position + camera_direction, camera_direction]  # Represents the plane the display is in [point, normal unit vector]
#Objects
class Triangle:
    def __init__(self, v1, v2, v3):  # Vertices of the triangle, (should be np.array of shape (3, 1))
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    @property
    def normal(self):
        return np.cross(self.v2-self.v1, self.v3-self.v1)

#Testing
triangles = []
tri1 = Triangle(np.array([4, 2, 2]), np.array([4, 2, 3]), np.array([4, 3, 2]))
tri2 = Triangle(np.array([4, 3, 3]), np.array([4, 2, 3]), np.array([4, 3, 2]))
tri3 = Triangle(np.array([4, 2, 2]), np.array([5, 2, 2]), np.array([4, 3, 2]))
tri4 = Triangle(np.array([5, 3, 2]), np.array([5, 2, 2]), np.array([4, 3, 2]))
triangles.append(tri1)
triangles.append(tri2)
triangles.append(tri3)
triangles.append(tri4)

print(camera_direction)

"""
#Rendering
while True:
    display.fill(BLACK)
    camera_screen = [camera_position + camera_direction, camera_direction]  # Represents the plane the display is in.

    for triangle in triangles:
        vertices = [triangle.v1, triangle.v2, triangle.v3]
        intersections = []
        for vertex in vertices:
            line_to_camera = [vertex, camera_position - vertex]
            intersection = intersectionOfLineAndPlane(line_to_camera, camera_screen)
            mapped_to = (intersection - camera_position - camera_direction)*800
            intersections.append((mapped_to[1], mapped_to[2]))
        #print(intersections)
        p.draw.polygon(display, WHITE, intersections)

    p.display.update()
"""
