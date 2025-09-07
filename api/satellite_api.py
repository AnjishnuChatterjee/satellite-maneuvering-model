# In SatelliteAPI class, update the get_satellite_telemetry method:
def get_satellite_telemetry(self, sat_id: int) -> Dict:
    """Get comprehensive telemetry data for a satellite"""
    satellite = self.get_satellite(sat_id)
    if not satellite:
        return {}
    
    # Ensure position and velocity are lists, not numpy arrays
    position = [float(x) for x in satellite.position]
    velocity = [float(x) for x in satellite.velocity]
    
    return {
        "basic_info": {
            "id": satellite.id,
            "name": satellite.name,
            "position": position,
            "velocity": velocity,
            "altitude": satellite.altitude,
            "fuel_remaining": satellite.fuel_remaining,
            "operational_status": satellite.operational_status,
            "last_contact": satellite.last_contact.isoformat(),
            "threat_level": satellite.threat_level,
            "maneuvering": satellite.maneuvering
        },
        "orbital_parameters": self._calculate_orbital_parameters(satellite),
        "system_health": self._get_system_health(satellite),
        "fuel_status": self._get_fuel_status(satellite)
    }