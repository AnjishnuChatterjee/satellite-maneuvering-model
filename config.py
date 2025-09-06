# Update config.py with IR detection and maneuvering parameters

from dataclasses import dataclass

@dataclass
class SimulationConfig:
    num_satellites: int = 3
    num_debris: int = 15
    simulation_speed: float = 1.0
    trail_length: int = 50
    update_frequency: int = 30
    ir_range: int = 1000  # km
    danger_threshold: int = 500  # km
    maneuver_speed: float = 1.0
    fuel_consumption: float = 0.1
    # Additional parameters for enhanced functionality
    maneuver_acceleration: float = 0.5  # m/s²
    collision_probability_threshold: float = 0.01  # 1%
    fuel_isp: int = 300  # seconds
    satellite_mass: int = 1000  # kg