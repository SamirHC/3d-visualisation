import asyncio
import pygame as p
import math
import numpy as np
import time

p.init()

# Display Settings
CAPTION = "3d"
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

display = p.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
p.display.set_caption(CAPTION)
p.event.set_grab(True)
p.mouse.set_visible(False)

#Clock
clock = p.time.Clock()
FPS = 60

# Colors
BLACK = p.Color(0, 0, 0)
WHITE = p.Color(255, 255, 255)
GRAY = p.Color(128, 128, 128)
RED = p.Color(255, 0, 0)
GREEN = p.Color(0, 255, 0)
BLUE = p.Color(0, 0, 255)

#Objects
class Line:
    def __init__(self, point: np.ndarray, vector: np.ndarray):
        self.point = point
        self.vector = vector

class Plane:
    def __init__(self, point: np.ndarray, normal: np.ndarray):
        self.point = point
        self.normal = normal

class Shape:
    def __init__(self, vertices: np.ndarray, color: p.Color):
        self.vertices = vertices
        self.color = color

    @property
    def center(self):
        return np.sum(self.vertices, axis=0)/len(self.vertices)

    def map_point_to_screen(self, point, camera_position, camera_screen, inverseAxisMatrix, shift, scale):
        line_to_camera = Line(point, camera_position - point)
        intersection_point = get_line_plane_intersection(line_to_camera, camera_screen)
        mapped_point = ((np.dot(inverseAxisMatrix, intersection_point - camera_position)+shift)*scale)[:-1]
        return mapped_point

    def map_vertices(self, camera_position, camera_screen, inverseAxisMatrix, shift, scale):
        self.mapped_vertices = []
        for vertex in self.vertices:
            self.mapped_vertices.append(self.map_point_to_screen(vertex, camera_position, camera_screen, inverseAxisMatrix, shift, scale))
        return self.mapped_vertices

    def draw(self, camera_position, camera_screen, inverseAxisMatrix, shift, scale):
        for mapped_vertex in self.map_vertices(camera_position, camera_screen, inverseAxisMatrix, shift, scale):
            if np.linalg.norm(mapped_vertex) > DISPLAY_WIDTH**2:  # Coordinates can't be too extreme
                return
        self.draw_method()

    def draw_method(self):
        p.draw.polygon(display, self.color, np.rint(self.mapped_vertices))

    def vector_from_camera(self, camera_position):
        return self.center - camera_position

    def distance_from_camera(self, camera_position, camera_direction):
        return np.linalg.norm(self.vector_from_camera(camera_position))*self.facing(camera_position, camera_direction)

    def facing(self, camera_position, camera_direction):
        return -np.sign(np.dot(self.vector_from_camera(camera_position), camera_direction))

    def is_front(self, camera_position, camera_direction):
        return True if self.facing(camera_position, camera_direction) == 1 else False

class Triangle(Shape):
    def __init__(self, vertices: np.ndarray, color=WHITE):
        super().__init__(vertices, color)
        assert vertices.shape == (3, 3)
        self.v1 = vertices[0]
        self.v2 = vertices[1]
        self.v3 = vertices[2]

    @property
    def normal(self):
        return np.cross(self.v2-self.v1, self.v3-self.v1)

class Segment(Shape):
    def __init__(self, vertices, color=WHITE):
        super().__init__(vertices, color)
        self.v1 = vertices[0]
        self.v2 = vertices[1]

    def draw_method(self):
        p.draw.line(display, self.color, *np.rint(self.mapped_vertices))

# Vector Calculations
def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

def get_angle_between_vectors(u: np.ndarray, v: np.ndarray):  # Returns angle between vectors measured in radians
    # THETA = arccos(u.v/||u||||v||) 
    return np.arccos(np.clip(np.dot(unit(u), unit(v)), -1.0, 1.0))

def get_line_plane_intersection(line: Line, plane: Plane) -> np.ndarray | None:  # Returns the point (if any) of intersection between a line and a plane
    """
    Line is given by points p such that: p = p0 + tv, where p0 is a fixed point, v fixed direction vector, t scalar.
    Plane is given by points q such that: n.(q - p1) = 0, where n is normal to plane, p1 fixed point.
    
    Result is given by: p = p0 + (n.(p1-p0) / n.v) v
    """
    p0 = line.point
    v = line.vector
    p1 = plane.point
    n = plane.normal
    
    ndotv = np.dot(n, v)   
    if np.isclose(ndotv, 0):
        return None
    
    t = np.dot(n, p1 - p0) / ndotv
    return p0 + t*v
    
# Camera
def get_rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    # Anticlockwise rotations
    c, s = np.cos(alpha), np.sin(alpha)
    alpha_rotation_matrix = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
        ])
    # Matrix required to rotate a plane by BETA in the xz plane
    c, s = np.cos(beta), np.sin(beta)
    beta_rotation_matrix = np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
        ])
    # Matrix required to rotate a plane by GAMMA in the yz plane
    c, s = np.cos(gamma), np.sin(gamma)
    gamma_rotation_matrix = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
        ])
    resultant_matrix = np.dot(np.dot(alpha_rotation_matrix, beta_rotation_matrix), gamma_rotation_matrix)
    return resultant_matrix
        
def get_camera_direction(alpha, beta, gamma):
    i = np.sin(alpha)*np.sin(gamma) - np.cos(alpha)*np.sin(beta)*np.cos(gamma)
    j = -np.cos(alpha)*np.sin(gamma) - np.sin(alpha)*np.sin(beta)*np.cos(gamma)
    k = np.cos(beta)*np.cos(gamma)
    return np.array([i, j, k])

def get_angles_from_mouse() -> tuple[float, float, float]:
    x, y = p.mouse.get_rel()
    alpha = 0
    beta = x / DISPLAY_WIDTH
    gamma = y / DISPLAY_HEIGHT
    return alpha, beta, gamma

def getRelativeAxes(key, camera_direction):
    axesDict = {}
    axesDict["forward"] = -camera_direction
    axesDict["forward"][1] = 0.
    axesDict["side"] = np.cross(camera_direction, np.array([0, 1, 0]))
    return axesDict[key]




#Testing
shapes = []
##for i in range(11):
##        line_x = Line(np.array([[0, 0, i], [10, 0, i]]), RED)
##        line_z = Line(np.array([[i, 0, 0], [i, 0, 10]]), BLUE)
##        shapes += [line_x, line_z]
shapes.append(Triangle(np.array([[1, 0, -10], [0, 0, -10], [0, 1, -10]]), RED))
shapes.append(Triangle(np.array([[1, 0, -10], [1, 1, -10], [0, 1, -10]])))
tri1 = Triangle(np.array([[1, 0, 5], [0, 0, 5], [0, 1, 5]]), BLUE)
tri2 = Triangle(np.array([[1, 0, 5], [1, 1, 5], [0, 1, 5]]), BLUE)
shapes.append(tri1)
shapes.append(tri2)
tri1 = Triangle(np.array([[1, 0, 6], [0, 0, 6], [0, 1, 6]]), GREEN)
tri2 = Triangle(np.array([[1, 0, 6], [1, 1, 6], [0, 1, 6]]), GREEN)
shapes.append(tri1)
shapes.append(tri2)
##tri1 = Triangle(np.array([[0, 0, 5], [0, 0, 6], [0, 1, 6]]), GRAY)
##tri2 = Triangle(np.array([[0, 1, 6], [0, 1, 5], [0, 0, 5]]), GRAY)
##shapes.append(tri1)
##shapes.append(tri2)
##tri1 = Triangle(np.array([[1, 0, 5], [1, 0, 6], [1, 1, 6]]), RED)
##tri2 = Triangle(np.array([[1, 1, 6], [1, 1, 5], [1, 0, 5]]), RED)
##shapes.append(tri1)
##shapes.append(tri2)

# Run
async def main():
    # Initial Camera Settings
    alpha = 0.  # Angle in the xy plane
    beta = 0.  # Angle in the xz plane
    gamma = 0.  # Angle in the yz plane
    x0 = 0.  # x coordinate of camera
    y0 = 0.  # y coordinate of camera
    z0 = 0.  # z coordinate of camera

    axisMatrix = get_rotation_matrix(alpha, beta, gamma)  # The orientation of the xyz axis relative to the camera
    inverseAxisMatrix = np.linalg.inv(axisMatrix)  # The inverse of the above,  used in order to map back to the display
    camera_position = np.array([x0, y0, z0])  # Initially at the origin, position given as a coordinate np.array
    camera_direction = get_camera_direction(alpha, beta, gamma)  # Direction is given as a unit vector np.array and is the normal to the camera_screen
    camera_screen = [camera_position + camera_direction, camera_direction]  # Represents the plane the display is in [point, normal unit vector]
    shift = np.array([0.5, 0.5, 0])
    scale = np.array([DISPLAY_WIDTH, DISPLAY_WIDTH, 0])

    running = True
    t0= time.time()
    while running:
        # Rendering
        display.fill(BLACK)
        camera_screen = Plane(camera_position + camera_direction, camera_direction)  # Represents the plane the display is in.
        axisMatrix = get_rotation_matrix(alpha, beta, gamma)
        inverseAxisMatrix = np.linalg.inv(axisMatrix)
        shapes.sort(key=lambda x: x.distance_from_camera(camera_position, camera_direction), reverse=True)
        for shape in shapes:
            if shape.is_front(camera_position, camera_direction):
                shape.draw(camera_position, camera_screen, inverseAxisMatrix, shift, scale)
            else:
                break
        display.blit(p.transform.flip(display, False, False), (0, 0))
        p.display.update()
        # Controls
        keys = p.key.get_pressed()
        forward_speed, side_speed = (0, 0)
        if keys[p.K_a]:
            side_speed = -3
        elif keys[p.K_d]:
            side_speed = 3
        if keys[p.K_w]:
            forward_speed = 3
        elif keys[p.K_s]:
            forward_speed = -3
        elif keys[p.K_p]:
            print(camera_position)
            print(camera_direction)
        elif keys[p.K_ESCAPE]:
            p.event.post(p.QUIT)
        # Animate
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        delta_angles = get_angles_from_mouse()
        alpha += delta_angles[0]
        beta += delta_angles[1]
        gamma += delta_angles[2]
        #beta = -t
        #gamma = -0.2
        #camera_position = np.array([5*math.sin(t), 1, 5 + 5*math.cos(t)])
        camera_direction = get_camera_direction(alpha, beta, gamma)
        camera_position += forward_speed*dt*getRelativeAxes("forward", camera_direction) + side_speed*dt*getRelativeAxes("side", camera_direction)
        # Misc
        for event in p.event.get():
            if event.type == p.QUIT:
                running = False
        
        clock.tick(FPS)
        await asyncio.sleep(0)

asyncio.run(main())