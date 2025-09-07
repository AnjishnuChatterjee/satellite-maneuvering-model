"""
Enhanced Debris Tracking API for orbital dashboard
"""
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

@dataclass
class DebrisObject:
    """Represents a space debris object"""
    id: int
    name: str
    object_type: str  # 'fragment', 'rocket_body', 'defunct_satellite'
    position: List[float]
    velocity: List[float]
    size_estimate: float  # meters
    radar_cross_section: float  # m²
    temperature: float  # for IR detection
    last_tracked: datetime
    confidence_level: float
    threat_assessment: str

class DebrisTracker:
    """Enhanced debris tracking and collision prediction system"""
    
    def __init__(self):
        self.debris_objects: Dict[int, DebrisObject] = {}
        self.logger = logging.getLogger(__name__)
        self.collision_predictions: List[Dict] = []
        self.initialize_debris_population()
    
    def initialize_debris_population(self):
        """Initialize a realistic debris population"""
        debris_types = [
            {"type": "fragment", "count": 15, "size_range": (0.1, 2.0)},
            {"type": "rocket_body", "count": 3, "size_range": (5.0, 15.0)},
            {"type": "defunct_satellite", "count": 2, "size_range": (2.0, 8.0)}
        ]
        
        debris_id = 1
        for debris_type in debris_types:
            for i in range(debris_type["count"]):
                self.add_debris_object(
                    debris_id=debris_id,
                    object_type=debris_type["type"],
                    size_range=debris_type["size_range"]
                )
                debris_id += 1
    
    def add_debris_object(self, debris_id: int, object_type: str, size_range: Tuple[float, float]):
        """Add a new debris object to tracking"""
        # Generate random orbital parameters
        earth_radius = 6371.0
        altitude = np.random.uniform(400, 2000)  # LEO debris
        orbital_radius = earth_radius + altitude
        
        # Random position on orbit
        angle = np.random.uniform(0, 2 * np.pi)
        inclination = np.random.uniform(-0.2, 0.2)  # Small inclination variation
        
        position = [
            orbital_radius * np.cos(angle),
            orbital_radius * np.sin(angle),
            orbital_radius * inclination
        ]
        
        # Calculate orbital velocity with some randomness for debris
        mu = 398600.4418
        base_velocity = np.sqrt(mu / orbital_radius)
        velocity_variation = np.random.uniform(0.9, 1.1)
        
        velocity = [
            -base_velocity * np.sin(angle) * velocity_variation,
            base_velocity * np.cos(angle) * velocity_variation,
            np.random.uniform(-0.1, 0.1)
        ]
        
        # Object characteristics
        size = np.random.uniform(*size_range)
        rcs = size ** 2 * np.pi  # Simplified RCS calculation
        temperature = np.random.uniform(0.3, 0.9)  # IR signature
        
        debris = DebrisObject(
            id=debris_id,
            name=f"{object_type.upper()}-{debris_id:03d}",
            object_type=object_type,
            position=position,
            velocity=velocity,
            size_estimate=size,
            radar_cross_section=rcs,
            temperature=temperature,
            last_tracked=datetime.now(),
            confidence_level=np.random.uniform(0.7, 0.95),
            threat_assessment="LOW"
        )
        
        self.debris_objects[debris_id] = debris
        self.logger.info(f"Added debris object {debris.name}")
    
    def update_debris_position(self, debris_id: int, position: List[float], velocity: List[float]):
        """Update debris position and velocity"""
        if debris_id in self.debris_objects:
            self.debris_objects[debris_id].position = position
            self.debris_objects[debris_id].velocity = velocity
            self.debris_objects[debris_id].last_tracked = datetime.now()
    
    def get_debris_object(self, debris_id: int) -> Optional[DebrisObject]:
        """Get debris object by ID"""
        return self.debris_objects.get(debris_id)
    
    def get_all_debris(self) -> List[DebrisObject]:
        """Get all tracked debris objects"""
        return list(self.debris_objects.values())
    
    def predict_collisions(self, satellite_positions: List[Dict], time_horizon_hours: float = 24.0) -> List[Dict]:
        """Predict potential collisions between satellites and debris"""
        predictions = []
        
        for sat_data in satellite_positions:
            sat_pos = np.array(sat_data['position'])
            sat_vel = np.array(sat_data.get('velocity', [0, 0, 0]))
            
            for debris in self.debris_objects.values():
                debris_pos = np.array(debris.position)
                debris_vel = np.array(debris.velocity)
                
                # Calculate closest approach
                collision_data = self._calculate_closest_approach(
                    sat_pos, sat_vel, debris_pos, debris_vel, time_horizon_hours
                )
                
                if collision_data['min_distance'] < 10.0:  # 10km threshold
                    collision_prob = self._calculate_collision_probability(collision_data)
                    
                    if collision_prob > 0.001:  # 0.1% threshold
                        predictions.append({
                            'satellite_id': sat_data['id'],
                            'debris_id': debris.id,
                            'debris_name': debris.name,
                            'time_to_closest_approach': collision_data['time_to_ca'],
                            'minimum_distance_km': collision_data['min_distance'],
                            'collision_probability': collision_prob,
                            'threat_level': self._assess_threat_level(collision_prob, collision_data['min_distance'])
                        })
        
        # Sort by threat level and time
        predictions.sort(key=lambda x: (x['collision_probability'], -x['minimum_distance_km']), reverse=True)
        self.collision_predictions = predictions
        return predictions
    
    def _calculate_closest_approach(self, sat_pos, sat_vel, debris_pos, debris_vel, time_horizon):
        """Calculate closest approach between satellite and debris"""
        rel_pos = debris_pos - sat_pos
        rel_vel = debris_vel - sat_vel
        
        # Time to closest approach
        if np.linalg.norm(rel_vel) < 1e-6:
            tca = 0
        else:
            tca = -np.dot(rel_pos, rel_vel) / np.dot(rel_vel, rel_vel)
        
        # Limit to time horizon
        tca = max(0, min(tca, time_horizon * 3600))
        
        # Positions at closest approach
        sat_pos_ca = sat_pos + sat_vel * tca
        debris_pos_ca = debris_pos + debris_vel * tca
        
        # Minimum distance
        min_distance = np.linalg.norm(sat_pos_ca - debris_pos_ca)
        
        return {
            'time_to_ca': tca / 3600,  # Convert to hours
            'min_distance': min_distance,
            'sat_pos_ca': sat_pos_ca,
            'debris_pos_ca': debris_pos_ca
        }
    
    def _calculate_collision_probability(self, collision_data):
        """Calculate collision probability based on distance and uncertainties"""
        min_distance = collision_data['min_distance']
        
        # Simplified probability model
        # In reality, this would use covariance matrices and Monte Carlo methods
        sigma = 1.0  # 1km position uncertainty
        prob = np.exp(-(min_distance**2) / (2 * sigma**2))
        
        return min(prob, 0.1)  # Cap at 10%
    
    def _assess_threat_level(self, probability, distance):
        """Assess threat level based on probability and distance"""
        if probability > 0.01 or distance < 1.0:
            return "HIGH"
        elif probability > 0.005 or distance < 5.0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_debris_statistics(self) -> Dict:
        """Get comprehensive debris statistics"""
        debris_list = self.get_all_debris()
        
        if not debris_list:
            return {}
        
        # Count by type
        type_counts = {}
        for debris in debris_list:
            type_counts[debris.object_type] = type_counts.get(debris.object_type, 0) + 1
        
        # Size distribution
        sizes = [debris.size_estimate for debris in debris_list]
        
        # Threat assessment
        threat_counts = {}
        for debris in debris_list:
            threat_counts[debris.threat_assessment] = threat_counts.get(debris.threat_assessment, 0) + 1
        
        # Tracking quality
        avg_confidence = np.mean([debris.confidence_level for debris in debris_list])
        
        return {
            'total_objects': len(debris_list),
            'objects_by_type': type_counts,
            'size_statistics': {
                'min_size_m': min(sizes),
                'max_size_m': max(sizes),
                'avg_size_m': np.mean(sizes),
                'median_size_m': np.median(sizes)
            },
            'threat_distribution': threat_counts,
            'tracking_quality': {
                'average_confidence': avg_confidence,
                'high_confidence_objects': sum(1 for d in debris_list if d.confidence_level > 0.9),
                'low_confidence_objects': sum(1 for d in debris_list if d.confidence_level < 0.7)
            },
            'active_collision_predictions': len(self.collision_predictions)
        }
    
    def get_high_risk_debris(self, risk_threshold: float = 0.005) -> List[DebrisObject]:
        """Get debris objects with high collision risk"""
        high_risk = []
        
        for prediction in self.collision_predictions:
            if prediction['collision_probability'] > risk_threshold:
                debris = self.get_debris_object(prediction['debris_id'])
                if debris and debris not in high_risk:
                    high_risk.append(debris)
        
        return high_risk
    
    def simulate_debris_evolution(self, time_step_hours: float = 1.0):
        """Simulate debris orbital evolution"""
        for debris in self.debris_objects.values():
            # Simple orbital propagation
            position = np.array(debris.position)
            velocity = np.array(debris.velocity)
            
            # Update position
            dt = time_step_hours * 3600  # Convert to seconds
            new_position = position + velocity * dt
            
            # Simple gravitational acceleration (circular orbit approximation)
            r = np.linalg.norm(new_position)
            mu = 398600.4418
            acc = -mu * new_position / (r**3)
            new_velocity = velocity + acc * dt
            
            debris.position = new_position.tolist()
            debris.velocity = new_velocity.tolist()
            debris.last_tracked = datetime.now()
