# Update simulation_math.py with IR detection and maneuvering functions

import numpy as np

EARTH_RADIUS_KM = 6371.0
MU = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
IR_DETECTION_RANGE = 1000  # km
DANGER_THRESHOLD = 500  # km

def circular_orbit_position(semi_major_axis, time, phase_offset=0.0, z_variation=0.0):
    """
    Calculate position in a simple circular orbit.
    """
    mean_motion = np.sqrt(MU / (semi_major_axis ** 3))  # radians/sec
    angle = mean_motion * time + phase_offset

    x = semi_major_axis * np.cos(angle)
    y = semi_major_axis * np.sin(angle)
    z = z_variation * np.sin(angle * 0.5)

    return [x, y, z]

def elliptical_orbit_position(semi_major_axis, time, phase_offset=0.0, eccentricity=0.1, z_amplitude=100):
    """
    Calculate position in an elliptical orbit with z-axis variation.
    """
    mean_motion = np.sqrt(MU / (semi_major_axis ** 3))
    angle = mean_motion * time + phase_offset

    r = semi_major_axis * (1 + eccentricity * np.sin(angle * 3))
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    z = z_amplitude * np.sin(angle * 2 + phase_offset)

    return [x, y, z]

def calculate_threat_level(position, frame_index, object_id):
    """
    Simulate threat level based on position and time.
    """
    base_threat = 0.1 + 0.3 * np.sin(frame_index * 0.01 + object_id)
    return max(0.0, min(1.0, base_threat))

def fuel_consumption(initial_fuel, frame_index, consumption_rate=0.1):
    """
    Simulate fuel depletion over time.
    """
    return max(0.0, initial_fuel - frame_index * consumption_rate)

def random_debris_orbit(debris_id, time, num_debris):
    """
    Generate randomized orbital parameters for debris.
    """
    rng = np.random.RandomState(debris_id)
    a = EARTH_RADIUS_KM + 600 + rng.uniform(0, 1500)
    phase = debris_id * 2 * np.pi / num_debris
    return elliptical_orbit_position(a, time, phase_offset=phase)

def get_debris_type(debris_id):
    """
    Cycle through predefined debris types.
    """
    return ['rocket_body', 'fragment', 'natural'][debris_id % 3]

# New functions for IR detection and maneuvering
def simulate_ir_detection(satellite_pos, debris_pos, debris_temperatures):
    """
    Simulate IR detection of debris by satellites.
    Returns detection probabilities for each debris object.
    """
    detection_probs = []
    
    for i, debris_pos in enumerate(debris_pos):
        distance = np.linalg.norm(np.array(satellite_pos) - np.array(debris_pos))
        
        # IR detection probability based on distance and debris temperature
        detection_prob = max(0, 1 - (distance / IR_DETECTION_RANGE)) * debris_temperatures[i]
        detection_probs.append(detection_prob)
    
    return detection_probs

def calculate_evasion_maneuver(satellite_pos, satellite_vel, threat_pos, threat_vel):
    """
    Calculate an evasion maneuver to avoid a threat.
    Returns the recommended maneuver vector.
    """
    # Calculate time to closest approach
    rel_pos = np.array(threat_pos) - np.array(satellite_pos)
    rel_vel = np.array(threat_vel) - np.array(satellite_vel)
    
    # If objects are moving in parallel, no approach
    if np.linalg.norm(rel_vel) < 1e-6:
        return [0, 0, 0]
    
    # Time to closest approach
    tca = -np.dot(rel_pos, rel_vel) / np.dot(rel_vel, rel_vel)
    
    # Position at closest approach
    sat_pos_tca = np.array(satellite_pos) + np.array(satellite_vel) * tca
    threat_pos_tca = np.array(threat_pos) + np.array(threat_vel) * tca
    
    # Distance at closest approach
    distance_tca = np.linalg.norm(sat_pos_tca - threat_pos_tca)
    
    # If distance is safe, no maneuver needed
    if distance_tca > DANGER_THRESHOLD:
        return [0, 0, 0]
    
    # Calculate evasion direction (perpendicular to velocity)
    evasion_dir = np.cross(satellite_vel, rel_pos)
    if np.linalg.norm(evasion_dir) < 1e-6:
        evasion_dir = np.cross(satellite_vel, [1, 0, 0])  # Fallback direction
    
    evasion_dir = evasion_dir / np.linalg.norm(evasion_dir)
    
    # Maneuver intensity based on threat level
    threat_level = max(0, 1 - (distance_tca / DANGER_THRESHOLD))
    maneuver_magnitude = threat_level * 0.1  # Adjust as needed
    
    return (evasion_dir * maneuver_magnitude).tolist()

def apply_maneuver(satellite_pos, satellite_vel, maneuver_vector, dt):
    """
    Apply a maneuver to a satellite's position and velocity.
    """
    new_pos = np.array(satellite_pos) + np.array(satellite_vel) * dt
    new_vel = np.array(satellite_vel) + np.array(maneuver_vector)
    
    return new_pos.tolist(), new_vel.tolist()