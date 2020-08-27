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
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Initial Camera Settings
alpha = 0  # Angle in the xy plane
beta = 0# Angle in the xz plane
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
        return np.array([0, 0, 0])
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
    gamma_rotation_matrix = np.array([
        [1, 0, 0],
        [0, math.cos(GAMMA), -math.sin(GAMMA)],
        [0, math.sin(GAMMA), math.cos(GAMMA)]
        ])  # Matrix required to rotate a plane by ALPHA in the yz plane
    resultant_matrix = np.dot(np.dot(alpha_rotation_matrix, beta_rotation_matrix), gamma_rotation_matrix)
    return resultant_matrix
        
def findCameraDirection(ALPHA, BETA, GAMMA):
    i = math.sin(ALPHA)*math.sin(GAMMA) - math.cos(ALPHA)*math.sin(BETA)*math.cos(GAMMA)
    j = -math.cos(ALPHA)*math.sin(GAMMA) - math.sin(ALPHA)*math.sin(BETA)*math.cos(GAMMA)
    k = math.cos(BETA)*math.cos(GAMMA)
    return np.array([i, j, k])

def getInverseAxisMatrix(ALPHA, BETA, GAMMA):
    resultant_matrix = np.linalg.inv(getAxisMatrix(ALPHA, BETA, GAMMA))
    return resultant_matrix

axisMatrix = getAxisMatrix(alpha, beta, gamma)  # The orientation of the xyz axis relative to the camera
inverseAxisMatrix = np.linalg.inv(axisMatrix)  # The inverse of the above,  used in order to map back to the display
camera_position = np.array([x0, y0, z0])  # Initially at the origin, position given as a coordinate np.array
camera_direction = findCameraDirection(alpha, beta, gamma)  # Direction is given as a unit vector np.array and is the normal to the camera_screen
camera_screen = [camera_position + camera_direction, camera_direction]  # Represents the plane the display is in [point, normal unit vector]
shift = np.array([0.5, 0.5, 0])
scale = np.array([display_width, display_width, 0])

#Objects
class Triangle:
    def __init__(self, v1, v2, v3, color=WHITE):  # Vertices of the triangle, (should be np.array of shape (3, 1))
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.vertices = np.array([v1, v2, v3])
        self.color = color

    @property
    def normal(self):
        return np.cross(self.v2-self.v1, self.v3-self.v1)

class Line:
    def __init__(self, v1, v2, color=WHITE):
        self.v1 = v1
        self.v2 = v2
        self.vertices = np.array([v1, v2])
        self.color = color

#Testing
shapes = []
for i in range(-10, 10, 2):
    for j in range(-10, 10, 2):
        line_x = Line(np.array([-10, i, j]), np.array([10, i, j]), RED)
        line_y = Line(np.array([i, -10, j]), np.array([i, 10, j]), GREEN)
        line_z = Line(np.array([i, j, -10]), np.array([i, j, 10]), BLUE)
        shapes += [line_x, line_y, line_z]
tri1 = Triangle(np.array([1, 0, 5]), np.array([0, 0, 5]), np.array([0, 1, 5]))
tri2 = Triangle(np.array([1, 0, 5]), np.array([1, 1, 5]), np.array([0, 1, 5]))
shapes.append(tri1)
shapes.append(tri2)
tri3 = Triangle(np.array([1, 0, 10]), np.array([0, 0, 10]), np.array([0, 1, 10]), GRAY)
tri4 = Triangle(np.array([1, 0, 10]), np.array([1, 1, 10]), np.array([0, 1, 10]), GRAY)
shapes.append(tri3)
shapes.append(tri4)

# Run
while True:
    # Rendering
    display.fill(BLACK)
    camera_screen = [camera_position + camera_direction, camera_direction]  # Represents the plane the display is in.
    axisMatrix = getAxisMatrix(alpha, beta, gamma)
    inverseAxisMatrix = np.linalg.inv(axisMatrix)
    for shape in shapes:
        mapped_vertices = []
        for vertex in shape.vertices:
            line_to_camera = [vertex, camera_position - vertex]
            intersection_point = intersectionOfLineAndPlane(line_to_camera, camera_screen)
            mapped_point = ((np.dot(inverseAxisMatrix, intersection_point - camera_position)+shift)*scale)[:-1]
            mapped_vertices.append(mapped_point)
        if len(mapped_vertices) == 2:
            if np.linalg.norm(mapped_vertices[0]) < display_width**2 and np.linalg.norm(mapped_vertices[1]) < display_width**2:
                p.draw.line(display, shape.color, *np.rint(mapped_vertices))
        else:
            p.draw.polygon(display, shape.color, np.rint(mapped_vertices))
    p.transform.flip(display, False, True)
    p.display.update()
    # Animate
    t = time.time()
    camera_position = np.array([10*math.sin(t), 0, 10 + 10*math.cos(t)])
    camera_direction = findCameraDirection(alpha, beta, gamma)
