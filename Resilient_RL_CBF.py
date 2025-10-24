import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.controllers import *
import numpy as np
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import logging
from scipy.interpolate import CubicSpline
from math import pi  # or import numpy as np and use np.pi
from scipy.linalg import cholesky, eigvalsh
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Configure Logging
#logging.basicConfig(level=logging.DEBUG,  # Set the log level
                    #format='%(asctime)s - %(levelname)s - %(message)s')  # Set the log format

# Configuration Constants
CONFIG = {
    # Total number of robots in the system.
    "TOTAL_ROBOTS": 10,
    
    # Standard deviation of noise added to control inputs.
    "CONTROL_NOISE_STD": 0.0000001,
    
    # Covariance matrix representing the process noise in the system.
    "PROCESS_NOISE_COVARIANCE": np.eye(3) * 0.0000001,  # 3x3 identity matrix scaled by 0.01.
    
    # Threshold for triggering events in the control strategy.
    "EVENT_TRIGGER_THRESHOLD": 0.01,
    
    # Small constant to avoid division by zero in calculations.
    "EPSILON": 1e-6,
    
    # Standard deviation of noise in range measurements.
    "RANGE_NOISE_STD": 0.005,
    
    # Standard deviation of noise in bearing measurements.
    "BEARING_NOISE_STD": 0.05,
    
    # Operational area limits for the x-coordinate.
    "x_limits": (-2, 2),
    
    # Operational area limits for the y-coordinate.
    "y_limits": (-2,2),
    
    # Dimensions of the target area for the robots.
    "TARGET_AREA": (2, 2),
    
    # Number of steps in the simulation or control loop.
    "num_steps": 320,

    # Time interval for each iteration of the control loop.
    "dt": 0.1,
    
    # Initialize the integral term for the PID controller (2D control).
    "integral_leader": np.zeros(2),
    "integral_follower": np.zeros(2),
    
    # Initialize the previous error for the PID controller (2D control).
    "previous_error_leader": np.zeros(2),
    "previous_error_follower": np.zeros(2),

    # Limit for control input to prevent excessive values.
    "CONTROL_INPUT_LIMIT": 20,

    # Minimum safe distance between robots to avoid collisions.
    "SAFE_DISTANCE": 0.001,  # Minimum distance robots should maintain to avoid collisions.

    # Gain for the control barrier function (CBF) to ensure collision avoidance.
    "cbf_gain": 0.5,

    # Index of the leader robot (typically the first robot).
    "LEADER_INDEX": 0,

    # Index of the follower robots (all other robots in the system).
    "FOLLOWER_INDICES": [i for i in range(1, 50)],  # Automatically set based on TOTAL_ROBOTS.

    # Desired trajectory for the leader to follow.
    # This can be dynamically updated during the simulation if necessary.
    "desired_trajectory": np.array([5.0, 5.0]),  # Example initial target position for the leader.
    
    # Placeholder for storing follower indices for easier referencing in follower control.
    "FOLLOWER_INDEX": 1,  # Default for follower; update dynamically during the simulation if needed.

    # Distance threshold for reaching a waypoint.
    "close_enough": 0.01, 

    # Regular Sensing range of robots
    "regular_sensing_range": 7,
    
    # Shadow Sensing range of robots
    "shadow_sensing_range": 7, 

    # distance between the wheel
    "b": 0.1,

    # Assume the initial position of the leader is known
    "initial_leader_position": np.array([-0.00, 0.00]),  # Replace with actual initial position if different

    "SENSING_THRESHOLD": 0.7,
    "COMMUNICATION_THRESHOLD": 0.7,
    "MAX_DOS_ROBOTS": 10,
    "MAX_FDI_MEASUREMENTS": 10,
    "ZONE_RADIUS": 0.3,  # Radius of the circular danger zone in 2x2 area
    "ZONE_LAYERS": 5,  # Number of layers in the danger zone
    "BASE_ATTACK_PROB": 0.2,  # Base probability at the outermost layer

    "LEARNING_RATE": 0.05,   # Learning rate for parameter updates
    "MIN_SPREAD": 1e-2,
    "batch_size": 35

}



# Global Variables for Attack Detection
consecutive_large_innovations = 0
attack_threshold = 10.0  # Threshold for detecting an attack
benign_threshold = 15.0   # Threshold for benign detection
steps_for_attack = 1     # Steps needed to confirm an attack
state = "normal"         # Initial state

# Initialize Arrays for Innovations and Adaptive Thresholds
previous_innovations = np.zeros((3, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]))  # Store innovations for each robot across all iterations
adaptive_thresholds = np.full((CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]), CONFIG["EVENT_TRIGGER_THRESHOLD"])  # Adaptive thresholds for each robot across all iterations

# Define Range Limits
x_range = CONFIG["x_limits"]
y_range = CONFIG["y_limits"]

# Generate Nearby Positions for Demonstration
nearby_positions = 0.1*np.ones((2, CONFIG["TOTAL_ROBOTS"]))  

# Measurement Noise Covariance
num_nearby = nearby_positions.shape[1]
MEASUREMENT_NOISE_COVARIANCE = np.diag(
    [CONFIG["RANGE_NOISE_STD"]**2] * (num_nearby ) + 
    [CONFIG["BEARING_NOISE_STD"]**2] * (num_nearby)
)

def generate_initial_positions(num_robots, x_range, y_range, initial_leader_position):
    """
    Generate initial positions for the robots in a grid pattern around the leader's initial position.

    Parameters:
        num_robots (int): Total number of robots (including the leader).
        x_range (tuple): Operational range for the x-coordinate (min, max).
        y_range (tuple): Operational range for the y-coordinate (min, max).
        initial_leader_position (np.ndarray): Initial position of the leader robot (x, y).
        grid_size (int): The number of rows and columns for the grid layout for followers.

    Returns:
        np.ndarray: An array of shape (3, num_robots) containing the initial positions and orientations.
    """

    grid_size = 3
    
    # Create an array to hold the initial positions of all robots
    initial_positions = np.zeros((3, num_robots))  # 3 rows for [x, y, theta]

    # Set the initial position and orientation of the leader robot (index 0)
    initial_positions[0, 0] = initial_leader_position[0] -0.45  # Leader's x position
    initial_positions[1, 0] = initial_leader_position[1]  # Leader's y position
    initial_positions[2, 0] = 0  # Leader's orientation set to 90 degrees (pi/2)

    # Calculate spacing for grid placement
    spacing_x = 0.4*(x_range[1] - x_range[0]) / (grid_size + 1)  # Space between each robot in x-direction
    spacing_y = 0.4*(y_range[1] - y_range[0]) / (grid_size + 1)  # Space between each robot in y-direction

    # Position the followers in a grid-like pattern
    follower_index = 1
    for row in range(grid_size):
        for col in range(grid_size):
            if follower_index < num_robots:
                # Calculate the follower's position based on grid layout
                initial_positions[0, follower_index] = initial_leader_position[0] + (col - grid_size // 2) * spacing_x - 1.15
                initial_positions[1, follower_index] = initial_leader_position[1] + (row - grid_size // 2) * spacing_y + 0.0
                initial_positions[2, follower_index] = 0  # Random orientation
                follower_index += 1

    return initial_positions


# Randomly Initialize Current Positions for Each UGV across num_steps

# Define initial positions for all robots
initial_positions = generate_initial_positions(CONFIG["TOTAL_ROBOTS"], x_range, y_range, CONFIG["initial_leader_position"])

# Define nearby positions deterministically
nearby_positions = initial_positions[:2, :]

# Define positions for multiple time steps deterministically
num_steps = CONFIG["num_steps"]
positions = np.zeros((2, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]))

# Populate positions with a systematic offset over time
positions[:, :, 0] = initial_positions[:2, :]


# Initialize Robotarium Environment
robotarium_env = robotarium.Robotarium(number_of_robots=CONFIG["TOTAL_ROBOTS"], show_figure=True)


# Initialize Predicted State Estimates and Covariance Matrices
x_hat_pred = np.zeros((3, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]))  # State vector: [x, y, theta]

# Initialize error covariance matrices for each robot and each time step
# P will have the shape (TOTAL_ROBOTS, 3, 3, num_steps)
P_pred = np.zeros((3, 3, CONFIG["TOTAL_ROBOTS"],  CONFIG["num_steps"]))  # For time-dependent covariance matrices

# Fill initial covariance matrices with identity matrices for the first time step
for i in range(CONFIG["TOTAL_ROBOTS"]):
    P_pred[:, :, i, 0] = 0.01*np.eye(3)  # Initial covariance for the first time step

# Initialize State Estimates and Covariance Matrices
x_hat = np.zeros((3, CONFIG["TOTAL_ROBOTS"], CONFIG["num_steps"]))  # State vector: [x, y, theta]
# Initialize error covariance matrices for each robot and each time step
# P will have the shape (TOTAL_ROBOTS, 3, 3, num_steps)
P = np.zeros((3, 3, CONFIG["TOTAL_ROBOTS"],  CONFIG["num_steps"]))  # For time-dependent covariance matrices

# Fill initial covariance matrices with identity matrices for the first time step
for i in range(CONFIG["TOTAL_ROBOTS"]):
    P[:, :, i, 0] = 0.01*np.eye(3)  # Initial covariance for the first time step



# Control inputs (initial values)
control_inputs = np.ones((2, CONFIG["TOTAL_ROBOTS"])) * 0.01  # 2D control inputs (v, omega)

# Initialize leader state
State = 0 



def generate_spline_waypoints(initial_position, num_waypoints=8):
    # Define control points for the spline
    control_points_x = np.linspace(initial_position[0]-0.5, 2.2, num_waypoints)
    
    # Generate control points for y using a smooth function (e.g., sine or cosine)
    # This creates a natural-looking path
    control_points_y = -0.4 * np.sin(2 * np.pi * control_points_x)  # Sine wave for smoothness

    # Create a cubic spline using the control points
    cs = CubicSpline(control_points_x, control_points_y)

    # Generate the waypoints from the spline
    x_waypoints = np.linspace(control_points_x[0], control_points_x[-1], num_waypoints * 6)  # More points for smoothness
    y_waypoints = cs(x_waypoints)

    # Ensure the waypoints stay within operational limits
    y_waypoints = np.clip(y_waypoints, -1, 1)

    return np.array([x_waypoints, y_waypoints])


def stanley_control(current_position, current_heading, waypoints, k=0.5):
    """
    Stanley Controller for path tracking.
    
    Parameters:
    - current_position: (2,) array of the leader's current x, y position
    - current_heading: current orientation of the leader robot (radians)
    - waypoints: (2, N) array of x, y coordinates of the path
    - k: gain for cross-track error correction

    Returns:
    - control_input: velocity and heading angle adjustment
    """
    x, y = current_position
    closest_point, min_dist = None, float('inf')
    closest_idx = 0

    # Find the closest point on the path
    for i, waypoint in enumerate(waypoints.T):
        distance = np.linalg.norm(waypoint - current_position)
        if distance < min_dist:
            min_dist = distance
            closest_point = waypoint
            closest_idx = i

    # Calculate the cross-track error
    cross_track_error = min_dist

    # Calculate the heading error
    path_direction = np.arctan2(
        waypoints[1, closest_idx + 1] - waypoints[1, closest_idx],
        waypoints[0, closest_idx + 1] - waypoints[0, closest_idx]
    )
    heading_error = path_direction - current_heading

    # Stanley control formula
    steering_angle = heading_error + np.arctan(k * cross_track_error)

    # Proportional speed control
    velocity = 300 * np.exp(-abs(steering_angle))  # Reduces speed if steering angle is large

    control_input = np.array([velocity, steering_angle])
    return control_input



def state_transition(previousPose, ut):
    # Estimates the next position and orientation of a 2 wheeled robot
    # INPUT: 
    # previousPose: 1x3 array [previousX, previousY, previousTheta]
    # ut: 1x2 array [DL, DR]
    
    previousX = previousPose[0]
    previousY = previousPose[1]
    previousTheta = previousPose[2]

    DL = ut[0]
    DR = ut[1]


    # Calculate the new x coordinate
    currentX = previousX + ((DR + DL) / 2) * np.cos(previousTheta + ((DR - DL) / (2 * CONFIG["b"])))

    # Calculate the new y coordinate
    currentY = previousY + ((DR + DL) / 2) * np.sin(previousTheta + ((DR - DL) / (2 * CONFIG["b"])))

    # Calculate the new Theta
    currentTheta = previousTheta + (DR - DL) / CONFIG["b"]

    # Normalize results between [-pi, pi]
    currentTheta = normalizeAngle(currentTheta)

    # Create the output as a numpy array
    return np.array([currentX, currentY, currentTheta]).flatten()

def normalizeAngle(originalAngle):
    """
    Normalizes angle to the range [-pi, pi].

    Args:
        originalAngle (float): Input angle in radians.

    Returns:
        float: Angle normalized to the range [-pi, pi].
    """
    # Normalize to [0, 2*pi)
    normalizedAngle = originalAngle % (2 * np.pi)

    # Shift to [-pi, pi] if necessary
    if normalizedAngle > np.pi:
        normalizedAngle -= 2 * np.pi

    return normalizedAngle




def apply_cbf_control(original_control_input, robot_position, nearby_positions, min_distance, adversary_center, adversary_spread, adversary_alpha, config, P_threshold=0.095):
    """
    Modifies the control input using Control Barrier Function to avoid collisions and adversarial regions.
    
    Parameters:
        original_control_input (np.ndarray): The original control input.
        robot_position (np.ndarray): Position of the robot (shape: (2,)).
        nearby_positions (np.ndarray): Positions of nearby robots (shape: (2, N)).
        min_distance (float): Minimum allowable distance to avoid collision.
        adversary_center (np.ndarray): The center of the adversarial region (shape: (2,)).
        adversary_spread (float): Spread parameter of the adversarial region.
        adversary_alpha (float): Maximum probability of an attack at the center of the adversarial region.
        config (dict): Configuration dictionary containing control parameters.
        
    Returns:
        np.ndarray: Modified control input, clipped to stay within specified limits.
    """
    
    # Calculate Control Barrier Function (CBF) for avoiding collisions with nearby robots
    h_ij = control_barrier_function(robot_position, nearby_positions, min_distance)

    # If the barrier function is violated (too close), adjust the control input
    if (h_ij < 0).any():
        # Compute the direction for repulsive force
        direction = (robot_position[:, np.newaxis] - nearby_positions)  # Shape (2, N)

        # Normalize the direction vector to avoid division by zero
        norm = np.linalg.norm(direction, axis=0) + 1e-6  # Shape (N,)
        normalized_direction = direction / norm  # Shape (2, N)

        # Calculate the repulsive force based on the normalized direction
        repulsive_force = config["cbf_gain"] * normalized_direction / norm  # Shape (2, N)

        # Sum the repulsive forces and apply to the original control input
        original_control_input += np.sum(repulsive_force, axis=1)
    
    # Incorporate the adversarial region (attack probability) into the control input
    dist_to_adversary = np.linalg.norm(robot_position - adversary_center)  # Distance to adversary region center
    prob_of_attack = adversary_alpha * np.exp(-dist_to_adversary**2 / (2 * adversary_spread**2))

    # If robot is near the adversarial region (probability exceeds a threshold), adjust control
    if prob_of_attack > P_threshold:
        # Compute direction away from the adversarial region center
        direction_from_adversary = robot_position - adversary_center
        norm_adv = np.linalg.norm(direction_from_adversary) + 1e-6  # Avoid division by zero
        normalized_direction_adv = direction_from_adversary / norm_adv  # Direction away from adversary

        # Apply a repulsive force based on adversary probability (stronger if probability is high)
        adversary_repulsive_force = config["cbf_gain"] * prob_of_attack * normalized_direction_adv / norm_adv
        
        # Adjust the control input to steer away from the adversarial region
        original_control_input += adversary_repulsive_force
    
    return original_control_input

def control_barrier_function(robot_position, nearby_positions, min_distance):
    """
    Computes the control barrier function values to avoid collisions with nearby robots.
    
    Parameters:
        robot_position (np.ndarray): Position of the robot (shape: (2,)).
        nearby_positions (np.ndarray): Positions of nearby robots (shape: (2, N)).
        min_distance (float): Minimum allowable distance to avoid collision.
        
    Returns:
        np.ndarray: Barrier function values (shape: (N,)).
    """
    # Calculate pairwise distances between the robot and nearby robots
    distances = np.linalg.norm(robot_position[:, np.newaxis] - nearby_positions, axis=0)
    
    # Barrier function: should be greater than 0 for safe distances, negative otherwise
    h_ij = min_distance - distances
    
    return h_ij


def leader_control_policy(current_position, current_heading, positions, min_distance, config, waypoints):
    """
    Control policy for the leader robot with Pure Pursuit and CBF for collision avoidance.
    
    Parameters:
        current_position (np.ndarray): Current position of the leader robot.
        target_position (np.ndarray): Target position for the leader robot.
        positions (np.ndarray): Positions of all robots (shape: (2, N)).
        min_distance (float): Minimum safe distance.
        config (dict): Configuration parameters.
        waypoints (np.ndarray): Waypoints for Pure Pursuit controller.
        
    Returns:
        np.ndarray: Control input for the leader robot.
    """
    # Pure Pursuit control to track the waypoints
    #control_input = pure_pursuit_control(current_position, target_position)
    control_input = stanley_control(current_position, current_heading, waypoints)

    
    return  control_input




def apply_process_noise(control_input, config):
    """
    Applies process noise to the control input.
    
    Parameters:
        control_input (np.ndarray): Control input before noise application.
        config (dict): Configuration parameters for noise application.
        
    Returns:
        np.ndarray: Noisy control input.
    """
    noise = np.random.normal(0, CONFIG["CONTROL_NOISE_STD"], size=control_input.shape)
    control_input_noisy = control_input + noise
    return control_input_noisy







def calculate_algebraic_connectivity(L):
    """
    Calculate the algebraic connectivity (second smallest eigenvalue) of the Laplacian matrix.

    Parameters:
    - L (np.ndarray): Laplacian matrix of the graph.

    Returns:
    - lambda_2 (float): Algebraic connectivity.
    """
    eigenvalues = eigvalsh(L)
    return eigenvalues[1]  # The second smallest eigenvalue





# Initialize a dictionary to store consecutive large innovation counts for each pair
consecutive_large_innovations = {}

def sensing_attack_detected(innovation, threshold=0.00, steps_for_attack=3):
    """
    Detects attacks in segments of the innovation array. Each segment (pair of elements)
    is checked to see if it consistently exceeds the threshold over a certain number of steps.
    
    Parameters:
    - innovation (np.ndarray): The innovation vector.
    - threshold (float): The threshold for attack detection.
    - steps_for_attack (int): Number of consecutive steps exceeding the threshold required to confirm an attack.
    
    Returns:
    - gamma (np.ndarray): An array indicating attack detection (1 for attack, 0 for no attack) for each segment.
    """
    
    # Calculate the number of pairs in the innovation array
    num_pairs = innovation.shape[0] // 2
    gamma_sensing = np.zeros(num_pairs, dtype=int)  # Initialize the gamma array
    
    # Loop through each pair in the innovation array
    for i in range(num_pairs):
        # Extract the i-th pair
        innovation_pair = innovation[2 * i: 2 * i + 2]
        
        # Calculate the norm of the pair to check against the threshold
        pair_norm = np.linalg.norm(innovation_pair)
        
        # Initialize consecutive count if not present in the dictionary
        if i not in consecutive_large_innovations:
            consecutive_large_innovations[i] = 0
            
        # Check if the norm of the pair exceeds the threshold
        if pair_norm > threshold:
            consecutive_large_innovations[i] += 1
        else:
            consecutive_large_innovations[i] = 0
        
        # Confirm an attack if the consecutive count exceeds the steps_for_attack
        if consecutive_large_innovations[i] >= steps_for_attack:
            gamma_sensing[i] = 1
        else:
            gamma_sensing[i] = 0
    # Calculate the sensing attack rate (D_s) as the percentage of robot pairs affected
    D_s = np.sum(gamma_sensing) / num_pairs  # Sensing attack rate
    
    return gamma_sensing

# Initialize a dictionary to store the last estimate for each robot and a counter for the number of steps without change
last_estimate = {}
steps_without_update = {}

# Initialize counters for DoS attacks
dos_attack_count = 0

def communication_attack_detected(current_estimates, update_threshold=20, steps_for_attack=3):
    """
    Detects communication (DoS) attacks based on the absence of updates in local estimates.
    A DoS attack is confirmed when local estimates do not change for a specified number of steps.

    Parameters:
    - current_estimates (np.ndarray): The current local estimates for each robot.
    - update_threshold (int): The number of steps after which a missing update is considered a DoS attack.
    - steps_for_attack (int): The number of consecutive steps exceeding the threshold required to confirm an attack.

    Returns:
    - D_c (float): The communication attack rate (fraction of robots affected by DoS attacks).
    """
    global last_estimate, steps_without_update, total_robots

    #print(current_estimates, "current_estimates")
    
    # Initialize an array to track DoS attack detection status
    gamma_comms = np.zeros(CONFIG["TOTAL_ROBOTS"], dtype=int)  # Communication (DoS) attack detection
    
    for robot_id, current_estimate in enumerate(current_estimates):
        # If this is the first update for this robot, initialize the last estimate and the counter
        if robot_id not in last_estimate:
            last_estimate[robot_id] = current_estimate
            steps_without_update[robot_id] = 0
            continue  # No attack detected yet for this robot
        
            # Compare the current estimate with the last estimate using np.array_equal to avoid ambiguity
        if np.array_equal(current_estimate, last_estimate[robot_id]):
            steps_without_update[robot_id] += 1
        else:
            # Reset the counter if the estimate has changed
            steps_without_update[robot_id] = 0

        
        # If steps without update exceed the threshold, a DoS attack is detected
        if steps_without_update[robot_id] >= update_threshold:
            gamma_comms[robot_id] = 1  # DoS attack detected for this robot
    
    # Calculate the communication attack rate (D_c) as the percentage of robots affected by DoS attacks
    D_c = np.sum(gamma_comms) / CONFIG["TOTAL_ROBOTS"]  # Communication attack rate
    
    return gamma_comms




# Assuming you have a function to initialize robots
def initialize_robots(r, leader_index, total_robots):
    # Colors: Red for leader, Green for followers
    colors = [np.array([1, 0, 0])] + [np.array([0, 1, 0])] * (total_robots - 1)
    
    # Initialize each robot with its respective color
    for i in range(total_robots):
        r.set_color(i, colors[i])


def determine_font_size(r, base_font_size):
    """
    Adjusts font size based on the environment size and robot count.

    Parameters:
    - r: Robotarium environment object, which may have attributes related to plotting dimensions.
    - base_font_size: The default or base font size to start from.

    Returns:
    - Adjusted font size.
    """
    # Adjust font size based on the number of robots or other factors
    adjusted_font_size = base_font_size
    
    # Example: if many robots, reduce font size for readability
    if r.number_of_robots > 10:  # Assumes r has an attribute for total robot count
        adjusted_font_size = max(8, base_font_size - 2)  # Reduce font size, minimum of 8
    
    return adjusted_font_size

def update_heading(current_heading, steering_angle):
    """
    Update the heading based on the steering angle and time delta.
    
    Parameters:
        current_heading (float): The current heading of the robot.
        steering_angle (float): The steering angle output from the control policy.
        dt (float): Time step for the simulation.
        
    Returns:
        float: Updated heading.
    """
    # Update the heading (ensure the heading stays within [0, 2*pi])
    new_heading = current_heading + steering_angle * CONFIG["dt"]
    new_heading = new_heading % (2 * np.pi)  # Wrap around if necessary
    return new_heading

def initialize_headings(num_robots, initial_leader_heading=0):
    """
    Initialize headings (orientations) for each robot in the team.

    Parameters:
        num_robots (int): Total number of robots (including the leader).
        initial_leader_heading (float): Initial heading angle of the leader in radians.

    Returns:
        np.ndarray: An array of shape (num_robots,) containing the initial headings.
    """
    # Initialize headings array
    headings = np.zeros(num_robots)

    # Set the leader's initial heading
    headings[0] = initial_leader_heading

    # Set initial headings for followers with fixed offsets
    headings[i] = initial_leader_heading 

    return headings

def create_laplacian_matrix(total_robots, ground_truth):
    """Create the Laplacian matrix based on ground_truth and sensing range."""
    
    # Initialize the adjacency matrix
    A = np.zeros((total_robots, total_robots))
    shadow = True

    # Fill the adjacency matrix based on the sensing range and shadow
    for i in range(total_robots):
        for j in range(total_robots):
            if i != j:  # Exclude self
                distance = np.linalg.norm(ground_truth[:, i] - ground_truth[:, j])
                
                # Regular neighbors based on regular sensing range
                if distance < CONFIG["regular_sensing_range"]:
                    A[i, j] = 1
                

    # Compute the degree matrix D
    D = np.diag(np.sum(A, axis=1))  # Sum along rows to get the degree of each node

    # Laplacian matrix L = D - A
    L = D - A

    return L

def topological_neighbors(L, agent, ground_truth):
    """ 
    Returns the regular and shadow neighbors of a particular agent using the graph Laplacian.
    
    L: NxN numpy array (representing the graph Laplacian)
    agent: int (agent: 0 - N-1)
    regular_sensing_range: float (distance for regular neighbors)
    shadow_sensing_range: float (distance for shadow neighbors)
    ground_truth: Nx3 numpy array (positions [x, y, theta] of robots)

    -> 1xM numpy arrays (regular_neighbors, shadow_neighbors)
    """
    # Validate the inputs
    assert isinstance(L, np.ndarray), f"In the topological_neighbors function, the graph Laplacian (L) must be a numpy ndarray. Received type {type(L).__name__}."
    assert isinstance(agent, int), f"In the topological_neighbors function, the agent number (agent) must be an integer. Received type {type(agent).__name__}."
    assert isinstance(CONFIG["regular_sensing_range"], (int, float)), "Sensing ranges must be numeric."
    assert isinstance(CONFIG["shadow_sensing_range"], (int, float)), "Sensing ranges must be numeric."
    
    # Validate agent index
    assert agent >= 0, f"In the topological_neighbors function, the agent number (agent) must be greater than or equal to zero. Received {agent}."
    assert agent < L.shape[0], f"In the topological_neighbors function, the agent number (agent) must be within the dimension of the provided Laplacian (L). Received agent number {agent} and Laplacian size {L.shape[0]} by {L.shape[1]}."

    # Extract the row corresponding to the agent
    row = L[agent, :]

    # Set the self-connection to zero (since the agent is not its own neighbor)
    row[agent] = 0

    # Initialize lists for regular and shadow neighbors
    regular_neighbors = []
    shadow_neighbors = []

    # Loop through all other agents to determine neighbors
    for j in range(L.shape[0]):
        if j != agent:
            # Calculate Euclidean distance between agents i and j
            dist = np.linalg.norm(ground_truth[:, agent] - ground_truth[:, j])

            # Regular neighbor: within the regular sensing range
            if dist < CONFIG["regular_sensing_range"]:
                regular_neighbors.append(j)
            # Shadow neighbor: within the shadow sensing range but outside regular range
            elif dist < CONFIG["shadow_sensing_range"]:
                shadow_neighbors.append(j)
                
    return np.array(regular_neighbors), np.array(shadow_neighbors)

# Function to recalculate the connectivity based on updated robot positions
def update_connectivity(ground_truth, config):
    # Initialize empty lists for connection indices
    rows = []
    cols = []
    
    # Recalculate Laplacian matrix based on current positions
    L = create_laplacian_matrix(config["TOTAL_ROBOTS"], ground_truth)

    # Populate rows and cols based on neighbors using the Laplacian matrix
    for i in range(config["TOTAL_ROBOTS"]):
        neighbors, _ = topological_neighbors(L, i, ground_truth)
        for neighbor in neighbors:
            rows.append(i)        # Current robot index
            cols.append(neighbor)  # Neighbor robot index

    # Convert rows and cols to numpy arrays for easier indexing
    return np.array(rows), np.array(cols)


# Function to update lines and labels for robots
def update_lines_and_labels(r, ground_truth, rows, cols, follower_indices, leader_index, line_leader, leader_label, line_followers, follower_labels):
    # Update leader position and line
    leader_label.set_position([ground_truth[0, leader_index], ground_truth[1, leader_index] + 0.15])
    leader_label.set_fontsize(12)
    
    # Ensure leader line is updated
    line_leader.set_data(
        [ground_truth[0, leader_index], ground_truth[0, follower_indices[0]]],
        [ground_truth[1, leader_index], ground_truth[1, follower_indices[0]]]
    )

    # Remove previous follower lines
    for line in line_followers:
        line.remove()
    line_followers.clear()

    # Update the lines for follower robots based on new connectivity
    for q, follower_index in enumerate(follower_indices):
        if follower_index >= ground_truth.shape[1]:
            continue

        follower_label = follower_labels[q]
        follower_label.set_position([ground_truth[0, follower_index], ground_truth[1, follower_index] + 0.15])
        follower_label.set_fontsize(12)
        
        # Create lines for each follower's updated connections
        for idx, neighbor in zip(rows, cols):
            if idx == follower_index:
                line_follower, = r.axes.plot(
                    [ground_truth[0, follower_index], ground_truth[0, neighbor]],
                    [ground_truth[1, follower_index], ground_truth[1, neighbor]],
                    linewidth=0.3, color='b', zorder=-1
                )
                line_followers.append(line_follower)  # Store reference to the line object


# Calculate distance from center
def calculate_distance(positions, center):
    return np.linalg.norm(positions - center, axis=1)

# Calculate adversary probability based on distance and layers
def adversary_probability(distance, radius, layers, base_prob):
    layer_width = radius / layers
    probabilities = np.zeros_like(distance)
    for i in range(layers):
        layer_start = i * layer_width
        layer_end = (i + 1) * layer_width
        prob = base_prob * (layers - i)  # Higher probability closer to center
        probabilities += np.where((distance >= layer_start) & (distance < layer_end), prob, 0)
    return probabilities


# Define the observation model function with adversary probability
def observation_model(Prob, P_threshold=0.1):
    # Assign label based on the probability threshold
    if Prob >= P_threshold:
        return 1  # Adversarial
    else:
        return 0  # Non-adversarial


def update_adversarial_regions_mini_batch(observations, center, spread, max_prob, batch_size = 5):
    """
    Update parameters of adversarial regions using mini-batch gradient descent.

    Args:
        observations (list of tuples): List of (observation, label) pairs.
        batch_size (int): Size of the mini-batch for gradient updates.
        center (float): Initial center parameter.
        spread (float): Initial spread parameter.
        max_prob (float): Initial maximum probability parameter.
        config (dict): Configuration dictionary containing hyperparameters like
                       MIN_SPREAD, LEARNING_RATE, and EPSILON.

    Returns:
        tuple: Updated center, spread, and max_prob parameters.
    """
    # Safeguard spread to avoid divide-by-zero errors
    min_spread = CONFIG["MIN_SPREAD"]
    spread = max(spread, min_spread)

    # Learning rate and epsilon for numerical stability
    learning_rate = CONFIG["LEARNING_RATE"]
    epsilon = CONFIG["EPSILON"]

    # Shuffle observations for mini-batch gradient descent
    np.random.shuffle(observations)

    # Process observations in mini-batches
    for i in range(0, len(observations), batch_size):
        batch = observations[i:i + batch_size]

        # Initialize gradients for the mini-batch
        grad_center = 0.0
        grad_spread = 0.0
        grad_max_prob = 0.0

        # Compute gradients for the current mini-batch
        for observation, label in batch:
            prob = max_prob * np.exp(-np.linalg.norm(observation - center)**2 / (2 * spread**2))

            grad_center += (label - prob) * (observation - center) / (spread**2)
            grad_spread += (label - prob) * (np.linalg.norm(observation - center)**2 - spread**2) / (spread**3)
            grad_max_prob += label / (max_prob + epsilon) - (1 - label) / (1 - max_prob + epsilon)

        # Average gradients over the mini-batch
        batch_size_actual = len(batch)  # Handle edge cases with smaller batches
        if batch_size_actual > 0:  # Avoid divide-by-zero errors
            grad_center /= batch_size_actual
            grad_spread /= batch_size_actual
            grad_max_prob /= batch_size_actual

        # Update adversarial parameters using gradient ascent/descent
        center += learning_rate * grad_center
        spread += learning_rate * grad_spread
        max_prob += learning_rate * grad_max_prob

        # Ensure parameters stay within valid ranges
        spread = max(spread, min_spread)  # Avoid zero/negative spread
        max_prob = np.clip(max_prob, 0.0, 1.0)  # Ensure probability stays in [0, 1]

    return center, spread, max_prob

    


# Define Neural Networks for RL Policy
def actor_network(state_dim, action_dim):
    return nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, action_dim),
        nn.Tanh()
    )

def critic_network(state_dim, action_dim):
    return nn.Sequential(
        nn.Linear(state_dim + action_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

# Initialize Networks
state_dim = 2  # Adapted for Robotarium environment (e.g., 2D position + velocity)
action_dim = 2  # Linear and angular velocity
actor = actor_network(state_dim, action_dim)
critic = critic_network(state_dim, action_dim)

# Select Action with Exploration (Add Noise)
def select_action_exploration(state, actor, noise=0.01):
    """
    Selects an action from the actor network with added noise for exploration.
    
    Parameters:
        state (np.ndarray): The current state of the robot.
        actor (nn.Module): The actor network that outputs the action given the state.
        noise (float): The standard deviation of the noise added for exploration.
    
    Returns:
        np.ndarray: The action selected by the actor with exploration noise added.
    """
    state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
    action = actor(state).detach().numpy()[0]  # Get the action from the actor network
    action += noise * np.random.randn(len(action))  # Add Gaussian noise for exploration
    return np.clip(action, -1, 1)  # Clip the action to the valid range of [-1, 1]

def select_action_exploitation(state, actor):
    """
    Selects the action from the actor network without exploration noise for exploitation.

    Parameters:
        state (np.ndarray): The current state of the robot.
        actor (nn.Module): The actor network that outputs the action given the state.

    Returns:
        np.ndarray: The action selected by the actor with no noise.
    """
    state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
    action = actor(state).detach().numpy()[0]  # Get the action from the actor network
    return np.clip(action, -1, 1)  # Clip the action to the valid range of [-1, 1]


# Reinforcement Learning Optimization using DDPG
def optimize_ddpg(actor, critic, memory, gamma=0.99, batch_size=32, lr_actor=0.0001, lr_critic=0.001):
    """
    Optimizes the Actor-Critic network for Deep Deterministic Policy Gradient (DDPG).
    
    Parameters:
        actor (nn.Module): The actor network that outputs actions given states.
        critic (nn.Module): The critic network that evaluates action-value pairs.
        memory (list): The replay buffer containing tuples of (state, action, reward, next_state, done).
        gamma (float): The discount factor for future rewards.
        batch_size (int): The number of samples per training batch.
        lr_actor (float): The learning rate for the actor network optimizer.
        lr_critic (float): The learning rate for the critic network optimizer.
    """
    # Ensure there is enough experience in the memory to sample a batch
    if len(memory) < batch_size:
        return
    
    # Sample a random batch from the memory
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # Convert data to tensors
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    # Optimize Critic
    # Get the predicted next action from the actor for the next states
    target_actions = actor(next_states).detach()
    # Calculate the target Q-value using the critic
    target_q_values = critic(torch.cat([next_states, target_actions], dim=1)).detach()
    # Calculate the expected Q-values for the current states and actions
    expected_q_values = rewards + (1 - dones) * gamma * target_q_values
    
    # Compute the current Q-values from the critic
    q_values = critic(torch.cat([states, actions], dim=1))
    # Compute the loss for the critic network
    critic_loss = nn.MSELoss()(q_values, expected_q_values)

    # Perform optimization on the critic network
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    # Optimize Actor
    # Compute the loss for the actor (maximize the Q-value given the predicted actions)
    actor_loss = -critic(torch.cat([states, actor(states)], dim=1)).mean()
    
    # Perform optimization on the actor network
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()



def exploration_policy(state, actor, adversary_center):
    """
    RL-based exploration policy for the Robotarium.
    Encourages circular exploration near the adversary center while following the RL-based action.
    
    Parameters:
        state (np.ndarray): Current state of the robot (position, velocity, etc.).
        actor (RL Actor): The reinforcement learning actor used to select actions.
        adversary_center (np.ndarray): The center of the adversarial region to explore around.

    Returns:
        np.ndarray: Combined action for the robot, balancing RL-based action and exploration around the adversary.
    """
    # Get the RL action (primary control input)
    rl_action = select_action_exploration(state, actor)

    print(rl_action, "rl_action")
    
    # Compute the direction towards the adversary center
    direction_to_center = adversary_center - state[:2]
    
    # Create a perpendicular direction to encourage circular motion around the adversary center
    perpendicular_direction = np.array([-direction_to_center[1], direction_to_center[0]])  # 90 degrees rotation
    perpendicular_direction /= np.linalg.norm(perpendicular_direction)  # Normalize
    
    # Combine the RL action with the exploration motion (encourage circular motion)
    exploration_action = rl_action + 50* perpendicular_direction  # 0.2 is the weight for exploration
    
    return exploration_action

def exploitation_policy(state, trajectory, actor, nearby_positions, min_distance, adversary_center, adversary_spread, adversary_alpha, config):
    """
    Computes the control input by balancing between following the desired trajectory,
    avoiding collisions with nearby robots, and steering away from adversarial regions.

    Parameters:
        state (np.ndarray): Current state of the robot (position, velocity, etc.).
        trajectory (np.ndarray): Desired trajectory to follow.
        actor (RL Actor): The reinforcement learning actor used to select actions.
        nearby_positions (np.ndarray): Positions of nearby robots.
        min_distance (float): Minimum allowable distance to avoid collisions.
        adversary_center (np.ndarray): The center of the adversarial region.
        adversary_spread (float): Spread parameter of the adversarial region.
        adversary_alpha (float): Maximum probability of an attack at the center.
        config (dict): Configuration dictionary containing control parameters.

    Returns:
        np.ndarray: Control input for the robot, considering safety and adversarial regions.
    """

    # Select the action from the RL actor
    rl_action = select_action_exploitation(state, actor) + 2 * (desired_trajectory - state[:2])
    
    # Apply the Control Barrier Function (CBF) to avoid collisions
    cbf_action = apply_cbf_control(rl_action, state[:2], nearby_positions, min_distance, adversary_center, adversary_spread, adversary_alpha, config)
    
    # Incorporate the desired trajectory to follow the leader, considering both safety and adversarial avoidance
    control_input = cbf_action   # Balance between safety and following leader
    
    return control_input



def should_switch_to_exploitation(state, target_state, exploration_threshold=0.8, max_exploration_steps=120, current_step=0):
    """
    Decides when to switch from exploration to exploitation based on distance to target and exploration step limits.
    
    Parameters:
        state (np.ndarray): Current state of the robot.
        target_state (np.ndarray): The target state the robot is trying to reach.
        exploration_threshold (float): Distance threshold for switching to exploitation (default 0.5).
        max_exploration_steps (int): Maximum number of exploration steps before switching to exploitation (default 1000).
        current_step (int): The current step number in the exploration phase.
    
    Returns:
        bool: True if the robot should switch to exploitation, False otherwise.
    """
    # Compute the distance to the target state (only consider position)
    distance_to_target = np.linalg.norm(state[:2] - target_state[:2])
    
    # If the robot is within the exploration threshold or exceeds the max exploration steps, switch to exploitation
    if distance_to_target < exploration_threshold:
        return True
    if current_step > max_exploration_steps:
        return True
    
    return False



def follower_control_policy(state, trajectory, actor, adversary_center, adversary_spread, adversary_alpha, config, target_state, current_step, min_distance):
    """
    Decides the control input for the robot based on the exploration or exploitation phase.
    
    Parameters:
        state (np.ndarray): Current state of the robot (position and possibly other state variables).
        leader_state (np.ndarray): State of the leader robot (for following in exploitation).
        actor (RL Actor): The RL actor used to select actions.
        adversary_model (object): The model of the adversary (used for exploration policy).
        adversary_center (np.ndarray): The center of the adversary region.
        beta (float): A weight parameter for blending exploration and exploitation actions.
        target_state (np.ndarray): The target state the robot is trying to reach.
        current_step (int): The current step in the exploration phase.
    
    Returns:
        np.ndarray: The control input, a combination of exploration and exploitation actions.
    """
    beta = 1
    # Check if the robot should switch to exploitation based on distance to target or exploration steps
    if should_switch_to_exploitation(state, target_state, exploration_threshold=0.5, max_exploration_steps=150, current_step=current_step):
        beta = 0  # Switch to full exploitation if conditions met
    
    # Compute the exploration and exploitation actions
    exploration_action = exploration_policy(state, actor, adversary_center)  # Exploration phase action
    exploitation_action = exploitation_policy(state, trajectory, actor, nearby_positions, min_distance, adversary_center, adversary_spread, adversary_alpha, config)  # Exploitation phase action
    # Combine the exploration and exploitation actions based on the value of beta
    # When beta is 1, we explore; when beta is 0, we exploit.
    control_input = beta * exploration_action + (1 - beta) * exploitation_action
    
    return control_input, beta





def run_simulation( CONFIG,  positions, topological_neighbors, 
                   generate_initial_positions):

    logging.info("Starting the simulation.")
    # Initialize tracking metrics for each robot over time
    adversarial_avoidance_over_time = np.zeros((CONFIG["num_steps"], CONFIG["TOTAL_ROBOTS"]))
    target_reach_status_over_time = np.zeros((CONFIG["num_steps"], CONFIG["TOTAL_ROBOTS"]))
    switch_parameter = np.zeros((CONFIG["num_steps"]))




    initial_positions = generate_initial_positions(CONFIG["TOTAL_ROBOTS"], x_range, y_range, CONFIG["initial_leader_position"])
    

    r = robotarium.Robotarium(number_of_robots=CONFIG["TOTAL_ROBOTS"],  initial_conditions=initial_positions, sim_in_real_time=True)
    _,uni_to_si_states = create_si_to_uni_mapping()
    si_to_uni_dyn = create_si_to_uni_dynamics()
    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
    leader_controller = create_si_position_controller(velocity_magnitude_limit=CONFIG["CONTROL_INPUT_LIMIT"])


    # For computational/memory reasons, initialize the velocity vector
    dxi = np.zeros((2,CONFIG["TOTAL_ROBOTS"]))


    leader_index = 0  # Leader is robot 0


    ground_truth = r.get_poses()


    leader_index = 0  # Leader is robot 0
    follower_indices = [i for i in range(1, CONFIG["TOTAL_ROBOTS"]) if i < ground_truth.shape[1]] 


    
    # Initialize empty lists for connection indices
    rows = []
    cols = []

    # Populate rows and cols based on neighbors
    for i in range(CONFIG["TOTAL_ROBOTS"]):
        L = create_laplacian_matrix(CONFIG["TOTAL_ROBOTS"], ground_truth)
        neighbors, _ = topological_neighbors(L, i, ground_truth)
        for neighbor in neighbors:
            rows.append(i)        # Current robot index
            cols.append(neighbor)  # Neighbor robot index

    # Convert rows and cols to numpy arrays for easier indexing
    rows = np.array(rows)
    cols = np.array(cols)

    # Initialize line objects and leader label
    line_leader, = r.axes.plot(
        [ground_truth[0, leader_index], ground_truth[0, follower_indices[0]]],
        [ground_truth[1, leader_index], ground_truth[1, follower_indices[0]]],
        linewidth=0.1, color='r', zorder=-1
    )

    leader_label = r.axes.text(
        ground_truth[0, leader_index], ground_truth[1, leader_index] + 0.15, "Leader",
        fontsize=12, color='k', fontweight='bold',
        horizontalalignment='center', verticalalignment='center', zorder=0
    )

    # Initialize follower lines and labels
    follower_labels = []
    line_followers = []  # Initialize a list to hold Line2D objects for followers

    for jj, follower_index in enumerate(follower_indices):
        if follower_index >= ground_truth.shape[1]:
            print(f"Follower index {follower_index} is out of bounds for x with size {ground_truth.shape[1]}")
            continue

        # Create lines for each follower's connections
        neighbors, _ =  topological_neighbors(L, i, ground_truth)
        for neighbor in neighbors:
            if neighbor < ground_truth.shape[1]:
                line_follower, = r.axes.plot(
                    [ground_truth[0, follower_index], ground_truth[0, neighbor]],
                    [ground_truth[1, follower_index], ground_truth[1, neighbor]],
                    linewidth=0.1, color='b', zorder=-1
                )
                line_followers.append(line_follower)  # Store reference to the line object

        # Add label for each follower
        follower_label = r.axes.text(
            ground_truth[0, follower_index], ground_truth[1, follower_index] + 0.15, f"Follower {jj + 1}",
            fontsize=12, color='b', fontweight='bold',
            horizontalalignment='center', verticalalignment='center', zorder=0
        )
        follower_labels.append(follower_label)

    # Mark the initial position
    initial_marker = r.axes.plot(CONFIG["initial_leader_position"][0]-0.3, CONFIG["initial_leader_position"][1], 'ro', label='Initial Position')
    r.axes.text(CONFIG["initial_leader_position"][0] -0.5, CONFIG["initial_leader_position"][1] + 0.05, 'Initial Position', 
             horizontalalignment='center', color='black')
    
    waypoints = generate_spline_waypoints(CONFIG["initial_leader_position"])
    d_min = CONFIG["SAFE_DISTANCE"]  # Minimum safe distance to avoid collisions



    # Additional plot settings if needed
    r.axes.set_xlim(-2, 2)
    r.axes.set_ylim(-2, 2)
    r.axes.set_title('Robot Path with Waypoints')
    r.axes.set_xlabel('X Coordinate')
    r.axes.set_ylabel('Y Coordinate')
    r.axes.axhline(0, color='black', linewidth=0.5, ls='--')
    r.axes.axvline(0, color='black', linewidth=0.5, ls='--')
    r.axes.grid()
    r.axes.legend()
    r.axes.legend(loc='upper left', bbox_to_anchor=(0.2, 0.2))



    r.step()  # Update Robotarium environment
    observations_comm = {}
    observations_sense = {}
    initial_alpha_comm = 0.0  # Scalar
    initial_alpha_sense = 0.0  # Scalar
    initial_center_comm = np.zeros(2)  # 2D center as an array
    initial_center_sense = np.zeros(2)  # 2D center as an array
    initial_sigma_comm = 0.0  # Scalar
    initial_sigma_sense = 0.0  # Scalar
    label_comm = 0
    label_sense = 0

        # Ensure observations_comm and observations_sense are lists of lists
    observations_comm = [None] * len(ground_truth[0])
    observations_sense = [None] * len(ground_truth[0])

        # Initialize adversarial region parameters for communication and sensing
    center_z_comm = {i: initial_center_comm for i in range(len(ground_truth[0]))}
    sigma_z_comm = {i: initial_sigma_comm for i in range(len(ground_truth[0]))}
    alpha_z_comm = {i: initial_alpha_comm for i in range(len(ground_truth[0]))}

    center_z_sense = {i: initial_center_sense for i in range(len(ground_truth[0]))}
    sigma_z_sense = {i: initial_sigma_sense for i in range(len(ground_truth[0]))}
    alpha_z_sense = {i: initial_alpha_sense for i in range(len(ground_truth[0]))}


        # Mini-batch observations storage
    batch_observations_comm = {i: [None] * CONFIG["batch_size"] for i in range(CONFIG["TOTAL_ROBOTS"])}  # For communication
    batch_observations_sense = {i: [None] * CONFIG["batch_size"] for i in range(CONFIG["TOTAL_ROBOTS"])}  # For sensing
        # Initialize lists to store the observations for each robot
    observations_comm_collected = {i: [] for i in range(CONFIG["TOTAL_ROBOTS"])}  # For communication
    observations_sense_collected = {i: [] for i in range(CONFIG["TOTAL_ROBOTS"])}  # For sensing
    # Initialize the memory (deque for efficient appending and popping)
    memory = deque(maxlen=10000)  # Set a limit on the memory size
    # Store initial positions to compute distance traveled
    previous_positions = np.copy(ground_truth[:2, :])  
    beta = 1

    # Simulation Loop
    for step in range(CONFIG["num_steps"]):  


        # Step the Robotarium environment
        ground_truth = r.get_poses()
        xi = uni_to_si_states(ground_truth)

        global Reference_Trajectory
        Reference_Trajectory = ground_truth

        #print(ground_truth, "ground_truth")

        initial_positions = generate_initial_positions(CONFIG["TOTAL_ROBOTS"], x_range, y_range, CONFIG["initial_leader_position"])



        rows, cols = update_connectivity(ground_truth, CONFIG)

            # Update lines and labels based on new connectivity
        update_lines_and_labels(r, ground_truth, rows, cols, follower_indices, leader_index, line_leader, leader_label, line_followers, follower_labels)

        leader_path = []
    
        # Get the current position of the leader robot
        leader_position = ground_truth[:, leader_index]  # Ensure this gets updated each step        
        # Update leader path by appending the current position
            # Update leader path by appending the current position (x and y only)
        leader_path = leader_position[:2]  # Store only x and y
        leader_path_marker = r.axes.plot(leader_path[0], leader_path[1], 'ro')
 
                # Danger zone centers
        center_comm = np.array([0.1, 0.5])  # Communication Danger Zone Center
        center_sense = np.array([10.2, 0.4])  # Sensing Danger Zone Center

                # Communication danger zone calculations
        ground_truth_2d = ground_truth[:2, :]
        distances_comm = calculate_distance(ground_truth_2d.T, center_comm)
        communication_probabilities = adversary_probability(distances_comm, CONFIG["ZONE_RADIUS"], CONFIG["ZONE_LAYERS"], CONFIG["BASE_ATTACK_PROB"])

        #print(communication_probabilities, "communication_probabilities")

        global D_c 
        D_c = np.mean(communication_probabilities)


        # Sensing danger zone calculations
        distances_sense = calculate_distance(ground_truth_2d.T, center_sense)
        sensing_probabilities = adversary_probability(distances_sense, CONFIG["ZONE_RADIUS"], CONFIG["ZONE_LAYERS"], CONFIG["BASE_ATTACK_PROB"])

        global D_s
        D_s = np.mean(sensing_probabilities)
            
                # DoS Attack (Communication Zone)
        dos_attacked_robots = []
        for i in range(CONFIG["TOTAL_ROBOTS"]):
            if np.random.rand() < communication_probabilities[i]:
                dos_attacked_robots.append(i)

        dos_attacked_robots = dos_attacked_robots[:CONFIG["MAX_DOS_ROBOTS"]]  # Limit DoS robots

        # FDI Attack (Sensing Zone)
        fdi_attacked_robots = []
        valid_indices = np.where(sensing_probabilities > 0)[0]
        if len(valid_indices) > 0:
            fdi_indices = np.random.choice(
                valid_indices, size=min(CONFIG["MAX_FDI_MEASUREMENTS"], len(valid_indices)), replace=False
            )
            for idx in fdi_indices:
                fdi_attacked_robots.append(idx)

        # Print results
        #print(f"FDI Attacked Robots: {fdi_attacked_robots}")
        #print(f"DoS Attacked Robots: {dos_attacked_robots}")
        


    
        for i in range(len(ground_truth[0])):
            # Get fresh observation for each robot at each step based on robot movement
            observation_comm = ground_truth[:2, i]  # This should change based on robot movement
            observation_sense = ground_truth[:2, i]  # This should change similarly

            label_comm = observation_model(communication_probabilities[i])
            label_sense = observation_model(sensing_probabilities[i])

            # Add observations to the batch for each robot
            batch_observations_comm[i][step % CONFIG["batch_size"]] = (observation_comm, label_comm)
            batch_observations_sense[i][step % CONFIG["batch_size"]] = (observation_sense, label_sense)
            
            observations_comm_collected[i].append(batch_observations_comm[i][step % CONFIG["batch_size"]])
            observations_sense_collected[i].append(batch_observations_sense[i][step % CONFIG["batch_size"]])

            L = create_laplacian_matrix(CONFIG["TOTAL_ROBOTS"], ground_truth)
            neighbors, shadow_neighbors = topological_neighbors(L, i, ground_truth)

            #print("neighbors", neighbors)
            
            if len(observations_comm_collected[0]) % CONFIG["batch_size"] == 0:  # Check if batch size is met
                # Update adversarial regions iteratively for communication
                center_z_comm[i], sigma_z_comm[i], alpha_z_comm[i] = update_adversarial_regions_mini_batch(
                    observations=observations_comm_collected[i],
                    center=center_z_comm[i],
                    spread=sigma_z_comm[i],
                    max_prob=alpha_z_comm[i],
                    batch_size=CONFIG["batch_size"],
                )

                # Update adversarial regions iteratively for sensing
                center_z_sense[i], sigma_z_sense[i], alpha_z_sense[i] = update_adversarial_regions_mini_batch(
                    observations=observations_sense_collected[i],
                    center=center_z_sense[i],
                    spread=sigma_z_sense[i],
                    max_prob=alpha_z_sense[i],
                    batch_size=CONFIG["batch_size"],
                )

                # Consensus: Update the adversarial regions using the neighbors' data
                sum_center_comm = center_z_comm[i]
                sum_sigma_comm = sigma_z_comm[i]
                sum_alpha_comm = alpha_z_comm[i]
                
                for j in neighbors:
                    sum_center_comm += center_z_comm[j]
                    sum_sigma_comm += sigma_z_comm[j]
                    sum_alpha_comm += alpha_z_comm[j]

                num_neighbors = len(neighbors)
                center_z_comm[i] = (1 / (num_neighbors + 1)) * sum_center_comm
                sigma_z_comm[i] = (1 / (num_neighbors + 1)) * sum_sigma_comm
                alpha_z_comm[i] = (1 / (num_neighbors + 1)) * sum_alpha_comm

                # Weighted update using w_i
                w_i = 0.06 # Adjust as needed
                center_z_comm[i] = (1 - w_i) * center_z_comm[i] + w_i * sum_center_comm
                sigma_z_comm[i] = np.clip((((1 - w_i) * sigma_z_comm[i] + w_i * sum_sigma_comm)), 0, 0.8)
                alpha_z_comm[i] = (1 - w_i) * alpha_z_comm[i] + w_i * sum_alpha_comm

                #print(f"Updated center_z_comm[{i}]: {center_z_comm[i]}")
                #print(f"Updated sigma_z_comm[{i}]: {sigma_z_comm[i]}")
                #print(f"Updated alpha_z_comm[{i}]: {alpha_z_comm[i]}")

                # Consensus for sensing parameters (similarly as for communication)
                sum_center_sense = center_z_sense[i]
                sum_sigma_sense = sigma_z_sense[i]
                sum_alpha_sense = alpha_z_sense[i]
                
                for j in neighbors:
                    sum_center_sense += center_z_sense[j]
                    sum_sigma_sense += sigma_z_sense[j]
                    sum_alpha_sense += alpha_z_sense[j]

                center_z_sense[i] = (1 / (num_neighbors + 1)) * sum_center_sense
                sigma_z_sense[i] = (1 / (num_neighbors + 1)) * sum_sigma_sense
                alpha_z_sense[i] = (1 / (num_neighbors + 1)) * sum_alpha_sense

                center_z_sense[i] = (1 - w_i) * center_z_sense[i] + w_i * sum_center_sense
                sigma_z_sense[i] = np.clip((((1 - w_i) * sigma_z_sense[i] + w_i * sum_sigma_sense)), 0, 0.8)
                alpha_z_sense[i] = (1 - w_i) * alpha_z_sense[i] + w_i * sum_alpha_sense




            # Sensing Danger Zone Layers
        sensing_colors = ['#FF6666', '#FF9999', '#FFCCCC', '#FFE6E6', '#FFB3B3']
        for i in range(CONFIG["ZONE_LAYERS"], 0, -1):
            radius = CONFIG["ZONE_RADIUS"] * (i / CONFIG["ZONE_LAYERS"])
            circle = plt.Circle(center_sense, radius,
                                color=sensing_colors[i - 1], alpha=0.01,
                                edgecolor='black', linewidth=0.1)
            r.axes.add_patch(circle)

        # Communication Danger Zone Layers
        communication_colors = ['#66B3FF', '#99CCFF', '#CCE5FF', '#E6F0FF', '#B3D9FF'] 
        for i in range(CONFIG["ZONE_LAYERS"], 0, -1):
            radius = CONFIG["ZONE_RADIUS"] * (i / CONFIG["ZONE_LAYERS"])
            circle = plt.Circle(center_comm, radius,
                                color=communication_colors[i - 1], alpha=0.01,
                                edgecolor='black', linewidth=0.1)
            r.axes.add_patch(circle)

        # Add text labels for both zones
        r.axes.text(center_sense[0], center_sense[1] + CONFIG["ZONE_RADIUS"] * 0.6, 
                'Sensing Danger Zone', fontsize=10, ha='center', color='blue')

        r.axes.text(center_comm[0], center_comm[1] + CONFIG["ZONE_RADIUS"] * 0.6, 
                'Communication Danger Zone', fontsize=10, ha='center', color='red')



        for i in range(CONFIG["TOTAL_ROBOTS"]):   

            
            L = create_laplacian_matrix(CONFIG["TOTAL_ROBOTS"], ground_truth)
            neighbors, shadow_neighbors = topological_neighbors(L, i, ground_truth)
            logging.info(f"Neighbors generated for robot {i}: {neighbors}")




            
        # Initialize state before the loop if it's not already initialized
            if "state_leader" not in locals():
                state_leader = 0  # Assuming 0 is the starting state or the initial index for waypoints

            headings_leader = 0.0
            # Leader control logic
            if i == CONFIG["LEADER_INDEX"]:
                leader_position = xi[:, CONFIG["LEADER_INDEX"]]
                target_position = waypoints[:,state_leader].reshape((2,))
                # Define and update the current heading of the leader robot
                current_heading = headings_leader
                control_input = leader_control_policy(leader_position, current_heading, xi, d_min, CONFIG, waypoints)
                global desired_trajectory
                desired_trajectory = control_input
                
                # Assuming control_input contains steering angle and velocity
                steering_angle = control_input[0]  # Extract steering angle from control input
                dt = CONFIG["dt"]  # Define time step for the simulation
        
                # Update the leader's heading
                headings_leader= update_heading(current_heading, steering_angle)

                # Check if the leader has reached its target waypoint
                if np.linalg.norm(leader_position - target_position) < CONFIG["close_enough"]:
                    state_leader = (state_leader + 1) % waypoints.shape[1]  # Update state to the next waypoint

            else:
                # Follower control logic
                follower_position = xi[:, i]  # Current position of the follower
                desired_trajectory = xi[:, CONFIG["LEADER_INDEX"]]  # Follower aims to follow the leader

                print(f"Updated center_z_comm[{i}]: {center_z_comm[i]}")
                print(f"Updated sigma_z_comm[{i}]: {sigma_z_comm[i]}")
                print(f"Updated alpha_z_comm[{i}]: {alpha_z_comm[i]}")

                target_position = [1.5, 0]

                # Call the follower control policy using Stanley Controller
                control_input, beta = follower_control_policy(follower_position[:2], desired_trajectory, actor, center_z_comm[i], sigma_z_comm[i], alpha_z_comm[i], CONFIG, target_position, step, d_min)
                


            # Apply process noise to the control input for both leader and followers
            dxi[:, i] = control_input
            if i == CONFIG["LEADER_INDEX"]:
                dxi[:, CONFIG["LEADER_INDEX"]] = control_input
            state = ground_truth[:2, i]
            action = dxi[:, i]  # Assuming action is velocity here
            reward = 0.0  # Define a reward function based on the simulation task
            next_state = state  # Assuming the state doesn't change much within one step for simplicity
            done = False  # For simplicity, you can set this to False, but you can define an episode-ending condition
            memory.append((state, action, reward, next_state, done))  # Store experience
            P_threshold=0.095
            Target_threshold = 0.1
            current_position = ground_truth[:2, i]  # Extract (x, y) position
            target_pos = np.array([1.5, 0])  # Ensure it's an array
            distance_to_target = np.linalg.norm(current_position - target_pos, axis=0)
            print(distance_to_target, "distance_to_target")

            # Compute probability of being in an adversarial zone
            is_in_adversarial_zone = communication_probabilities[i] 

            
            
            # Store Adversarial Avoidance: 1 if outside adversarial zone, 0 if inside
            adversarial_avoidance_over_time[step, i] = 1 - is_in_adversarial_zone

            switch_parameter[step] = beta

            print(switch_parameter[step], "switch_parameter[step]")



            # Store Target Reaching: 1 if within target threshold, 0 otherwise
            target_reach_status_over_time[step, i] = distance_to_target
            print(target_reach_status_over_time[step, i], "target_reach_status_over_time[step, i]")
            


            


        # Optimize DDPG (train the actor and critic networks)
        optimize_ddpg(actor, critic, memory)

        #print(dxi, "dxi")
        norms = np.linalg.norm(dxi, 20, 0)
        magnitude_limit = 20
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]
        # Apply control barriers, convert to unicycle, and set velocities
        dxi = si_barrier_cert(dxi, ground_truth[:2, :])
        dxu = si_to_uni_dyn(dxi, ground_truth)
        ground_truth_positions = []

        
        r.set_velocities(np.arange(CONFIG["TOTAL_ROBOTS"]), dxu)  # Update velocities for all robots
            # Step the simulation forward

        r.step()
    

   
    # Finalizing the simulation
    logging.debug(f"Final step {step}: current positions {positions}")
    avg_adversarial_avoidance = np.mean(adversarial_avoidance_over_time, axis=1)
    avg_target_reach = np.mean(target_reach_status_over_time, axis=1)
    print(avg_target_reach, "avg_target_reach")
    print(avg_adversarial_avoidance, "avg_adversarial_avoidance")

    # Create the time vector
    time_steps = np.arange(CONFIG["num_steps"])
    print(time_steps, "time_steps")

    control_input_data = []
    center_z_sense_data = []
    sigma_z_sense_data = []
    alpha_z_sense_data = []
    center_z_comm_data = []
    sigma_z_comm_data = []
    alpha_z_comm_data = []
    ground_truth_data = []

    # Collecting data during the simulation
    for step in range(num_steps):
        for i in range(CONFIG["TOTAL_ROBOTS"]):
            control_input_data.append([step, i, dxi[:, i][0], dxi[:, i][1]])
            center_z_sense_data.append([step, i] + list(center_z_sense[i]))
            sigma_z_sense_data.append([step, i, sigma_z_sense[i]])
            alpha_z_sense_data.append([step, i, alpha_z_sense[i]])
            center_z_comm_data.append([step, i] + list(center_z_comm[i]))
            sigma_z_comm_data.append([step, i, sigma_z_comm[i]])
            alpha_z_comm_data.append([step, i, alpha_z_comm[i]])
            ground_truth_data.append([step, i] + list(ground_truth[:2, i]))

    # Convert data to DataFrame
    df_control = pd.DataFrame(control_input_data, columns=["Step", "Robot", "v", "omega"])
    df_sense = pd.DataFrame(center_z_sense_data, columns=["Step", "Robot", "Center_x", "Center_y"])
    df_sense["Sigma"] = [row[2] for row in sigma_z_sense_data]
    df_sense["Alpha"] = [row[2] for row in alpha_z_sense_data]
    df_comm = pd.DataFrame(center_z_comm_data, columns=["Step", "Robot", "Center_x", "Center_y"])
    df_comm["Sigma"] = [row[2] for row in sigma_z_comm_data]
    df_comm["Alpha"] = [row[2] for row in alpha_z_comm_data]
    df_ground_truth = pd.DataFrame(ground_truth_data, columns=["Step", "Robot", "X", "Y"])

    # Save to CSV
    df_control.to_csv("control_input.csv", index=False)
    df_sense.to_csv("sensing_data.csv", index=False)
    df_comm.to_csv("communication_data.csv", index=False)
    df_ground_truth.to_csv("ground_truth.csv", index=False)

    
    

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()



        # Save data to CSV
    data = {
        "Time Step": time_steps,
        "Adversarial Avoidance": avg_adversarial_avoidance,
        "Target Tracking Error": avg_target_reach,
        "Switching Parameter": switch_parameter,
    }

    df = pd.DataFrame(data)
    csv_filename = "simulation_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data recorded in {csv_filename}")


    # Plot Adversarial Avoidance
    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, avg_adversarial_avoidance, label="Adversarial Avoidance", color="red", marker='o')
    plt.xlabel("Time Step")
    plt.ylabel("Adversarial Avoidance")
    plt.title("Adversarial Avoidance Over Time")
    plt.ylim(0, 1.2)
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, avg_target_reach, label="Target Reaching", color="green", marker='o')
    plt.xlabel("Time Step")
    plt.ylabel("Target Tracking Error")
    plt.title("Target Tracking Error Over Time")
    plt.ylim(0, 3)  # Ensure correct scale
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(time_steps, switch_parameter, label="switch_parameter", color="blue", marker='o')
    plt.xlabel("Time Step")
    plt.ylabel("switching parameter")
    plt.title("switching parameter between exploration action and explotation action")
    plt.ylim(0, 1.2)
    plt.legend()
    plt.grid()
    plt.show()

    input("Press Enter to close...")  # Pause until user input
    # Clean up the Robotarium environment
    logging.info("Simulation completed.")
    #Call at end of script to print debug information and for your script to run on the Robotarium server properly
    r.call_at_scripts_end()





# Call the run_simulation function to start
if __name__ == "__main__":
    # Define distances, errors, and rho before calling run_simulation
    # Initialize distances (e.g., with a placeholder array or a calculation)
    distances = np.zeros((CONFIG["TOTAL_ROBOTS"], CONFIG["TOTAL_ROBOTS"]))
    
    # Initialize errors 
    errors = np.zeros((CONFIG["TOTAL_ROBOTS"], CONFIG["TOTAL_ROBOTS"]))

    # Define rho
    rho = np.zeros(CONFIG["TOTAL_ROBOTS"])

    # Call the run_simulation function to start
if __name__ == "__main__":
    run_simulation(
        CONFIG=CONFIG,
        positions=positions,
        topological_neighbors=topological_neighbors,
        generate_initial_positions=generate_initial_positions,
    )

    

