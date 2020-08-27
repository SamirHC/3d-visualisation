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

# Vectors
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
        return np.array([256, 256, 256])
    w = P0 - V0
    s = -np.dot(n, w) / np.dot(n, u)  # Calculates the value of the parametric variable
    return P0 + s*u
    
# Camera
camera_position = np.array([0, 0, 0])  # Initially at the origin, position given as a coordinate
camera_direction = np.array([1, 0, 0])  # Initially looking parallel to the positive x-axis, direction is given as a unit vector
camera_screen = [camera_position + camera_direction, camera_direction]  # Represents the plane the display is in.
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

while True:
    display.fill(BLACK)
    camera_screen = [camera_position + camera_direction, camera_direction]  # Represents the plane the display is in.

    #Rendering
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
    camera_position = np.array([math.sin(time.time()), math.sin(0.5*time.time()), math.sin(0.25*time.time())])
    camera_direction = np.array([math.sin(time.time()), 0.25*math.sin(0.5*time.time()), 0.25*math.sin(0.25*time.time())])
    
