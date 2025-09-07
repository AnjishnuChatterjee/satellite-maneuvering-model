"""
Ground Station API for orbital dashboard
Manages ground station communications, data downlinks, and command uplinks
"""
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json

@dataclass
class GroundStation:
    """Represents a ground station"""
    id: int
    name: str
    location: List[float]  # [latitude, longitude, altitude]
    frequency_bands: List[str]
    max_elevation_angle: float
    operational_status: str
    last_contact: datetime
    data_rate_mbps: float

@dataclass
class CommunicationPass:
    """Represents a communication pass between satellite and ground station"""
    id: int
    satellite_id: int
    ground_station_id: int
    start_time: datetime
    end_time: datetime
    max_elevation: float
    data_volume_mb: float
    pass_quality: str
    status: str

class GroundStationAPI:
    """Ground station management and communication scheduling"""
    
    def __init__(self):
        self.ground_stations: Dict[int, GroundStation] = {}
        self.communication_passes: Dict[int, CommunicationPass] = {}
        self.logger = logging.getLogger(__name__)
        self.pass_counter = 1
        self.initialize_ground_stations()
    
    def initialize_ground_stations(self):
        """Initialize default ground stations"""
        stations = [
            {
                "id": 1, "name": "Kourou", "location": [5.2362, -52.7683, 0.05],
                "bands": ["S", "X", "Ka"], "data_rate": 150.0
            },
            {
                "id": 2, "name": "Madrid", "location": [40.4278, -4.2486, 0.67],
                "bands": ["S", "X"], "data_rate": 100.0
            },
            {
                "id": 3, "name": "Canberra", "location": [-35.4014, 149.0638, 0.69],
                "bands": ["S", "X", "Ka"], "data_rate": 120.0
            },
            {
                "id": 4, "name": "Goldstone", "location": [35.4267, -116.8900, 1.04],
                "bands": ["S", "X", "Ka"], "data_rate": 200.0
            }
        ]
        
        for station_data in stations:
            station = GroundStation(
                id=station_data["id"],
                name=station_data["name"],
                location=station_data["location"],
                frequency_bands=station_data["bands"],
                max_elevation_angle=85.0,
                operational_status="operational",
                last_contact=datetime.now() - timedelta(hours=np.random.randint(1, 12)),
                data_rate_mbps=station_data["data_rate"]
            )
            self.ground_stations[station_data["id"]] = station
    
    def calculate_satellite_visibility(self, satellite_position: List[float], 
                                     ground_station_id: int) -> Dict:
        """Calculate if satellite is visible from ground station"""
        if ground_station_id not in self.ground_stations:
            return {"visible": False, "elevation": 0, "azimuth": 0}
        
        station = self.ground_stations[ground_station_id]
        
        # Simplified visibility calculation
        # Convert ground station location to ECEF coordinates
        lat, lon, alt = station.location
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Earth radius + altitude
        earth_radius = 6371.0
        station_radius = earth_radius + alt
        
        # Ground station position in ECEF
        station_x = station_radius * np.cos(lat_rad) * np.cos(lon_rad)
        station_y = station_radius * np.cos(lat_rad) * np.sin(lon_rad)
        station_z = station_radius * np.sin(lat_rad)
        station_pos = np.array([station_x, station_y, station_z])
        
        # Satellite position
        sat_pos = np.array(satellite_position)
        
        # Vector from station to satellite
        sat_vector = sat_pos - station_pos
        distance = np.linalg.norm(sat_vector)
        
        # Calculate elevation angle (simplified)
        # Local vertical at station
        local_vertical = station_pos / np.linalg.norm(station_pos)
        
        # Elevation angle
        elevation_rad = np.arcsin(np.dot(sat_vector, local_vertical) / distance)
        elevation_deg = np.degrees(elevation_rad)
        
        # Azimuth calculation (simplified)
        azimuth_deg = np.degrees(np.arctan2(sat_vector[1], sat_vector[0]))
        
        # Visibility check
        visible = elevation_deg > 10.0  # Minimum elevation threshold
        
        return {
            "visible": visible,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "distance_km": distance,
            "signal_strength": max(0, 1 - distance / 2000)  # Simplified
        }
    
    def schedule_communication_pass(self, satellite_id: int, 
                                  ground_station_id: int,
                                  start_time: datetime,
                                  duration_minutes: int) -> int:
        """Schedule a communication pass"""
        pass_id = self.pass_counter
        self.pass_counter += 1
        
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Estimate data volume based on duration and station capability
        station = self.ground_stations[ground_station_id]
        data_volume = station.data_rate_mbps * duration_minutes * 60 / 8  # MB
        
        comm_pass = CommunicationPass(
            id=pass_id,
            satellite_id=satellite_id,
            ground_station_id=ground_station_id,
            start_time=start_time,
            end_time=end_time,
            max_elevation=45.0 + np.random.uniform(-15, 15),  # Simulated
            data_volume_mb=data_volume,
            pass_quality="good",
            status="scheduled"
        )
        
        self.communication_passes[pass_id] = comm_pass
        self.logger.info(f"Scheduled communication pass {pass_id}")
        return pass_id
    
    def get_upcoming_passes(self, hours_ahead: int = 24) -> List[Dict]:
        """Get upcoming communication passes"""
        current_time = datetime.now()
        future_time = current_time + timedelta(hours=hours_ahead)
        
        upcoming = []
        for pass_obj in self.communication_passes.values():
            if current_time <= pass_obj.start_time <= future_time:
                upcoming.append({
                    "id": pass_obj.id,
                    "satellite_id": pass_obj.satellite_id,
                    "ground_station": self.ground_stations[pass_obj.ground_station_id].name,
                    "start_time": pass_obj.start_time,
                    "duration_minutes": (pass_obj.end_time - pass_obj.start_time).total_seconds() / 60,
                    "max_elevation": pass_obj.max_elevation,
                    "data_volume_mb": pass_obj.data_volume_mb,
                    "status": pass_obj.status
                })
        
        return sorted(upcoming, key=lambda x: x["start_time"])
    
    def execute_pass(self, pass_id: int) -> bool:
        """Execute a communication pass"""
        if pass_id not in self.communication_passes:
            return False
        
        comm_pass = self.communication_passes[pass_id]
        comm_pass.status = "executing"
        
        # Update ground station last contact
        station = self.ground_stations[comm_pass.ground_station_id]
        station.last_contact = datetime.now()
        
        self.logger.info(f"Executing communication pass {pass_id}")
        return True
    
    def get_ground_station_status(self) -> List[Dict]:
        """Get status of all ground stations"""
        status_list = []
        
        for station in self.ground_stations.values():
            # Calculate time since last contact
            time_since_contact = datetime.now() - station.last_contact
            hours_since_contact = time_since_contact.total_seconds() / 3600
            
            status_list.append({
                "id": station.id,
                "name": station.name,
                "location": station.location,
                "status": station.operational_status,
                "hours_since_contact": round(hours_since_contact, 1),
                "data_rate_mbps": station.data_rate_mbps,
                "frequency_bands": station.frequency_bands
            })
        
        return status_list
    
    def get_communication_statistics(self) -> Dict:
        """Get communication statistics"""
        total_passes = len(self.communication_passes)
        completed_passes = sum(1 for p in self.communication_passes.values() 
                             if p.status == "completed")
        
        total_data_mb = sum(p.data_volume_mb for p in self.communication_passes.values())
        
        return {
            "total_passes": total_passes,
            "completed_passes": completed_passes,
            "success_rate": (completed_passes / max(1, total_passes)) * 100,
            "total_data_downlinked_mb": total_data_mb,
            "average_pass_duration": 8.5,  # minutes
            "active_ground_stations": len([s for s in self.ground_stations.values() 
                                         if s.operational_status == "operational"])
        }
    
    def predict_next_passes(self, satellite_id: int, hours_ahead: int = 48) -> List[Dict]:
        """Predict next communication opportunities"""
        predictions = []
        
        # Simplified prediction - generate passes every 90-120 minutes
        current_time = datetime.now()
        
        for i in range(int(hours_ahead * 60 / 100)):  # Approximate number of orbits
            pass_time = current_time + timedelta(minutes=i * 100 + np.random.randint(-10, 10))
            
            # Select random ground station
            station_id = np.random.choice(list(self.ground_stations.keys()))
            station = self.ground_stations[station_id]
            
            predictions.append({
                "satellite_id": satellite_id,
                "ground_station": station.name,
                "start_time": pass_time,
                "duration_minutes": np.random.randint(5, 15),
                "max_elevation": np.random.uniform(15, 85),
                "predicted_quality": np.random.choice(["excellent", "good", "fair"])
            })
        
        return sorted(predictions, key=lambda x: x["start_time"])[:10]  # Return top 10
