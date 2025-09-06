"""
Anomaly Detection API for orbital dashboard
Detects anomalies in satellite behavior, orbital parameters, and system health
"""
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json

@dataclass
class Anomaly:
    """Represents a detected anomaly"""
    id: int
    satellite_id: int
    anomaly_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    detected_at: datetime
    parameters_affected: List[str]
    confidence_score: float
    recommended_action: str
    status: str  # 'new', 'investigating', 'resolved', 'false_positive'

@dataclass
class HealthMetrics:
    """Satellite health metrics"""
    satellite_id: int
    timestamp: datetime
    battery_voltage: float
    solar_panel_current: float
    temperature: float
    attitude_error: float
    communication_strength: float
    thruster_pressure: float
    fuel_level: float

class AnomalyDetector:
    """Advanced anomaly detection for satellite systems"""
    
    def __init__(self):
        self.anomalies: Dict[int, Anomaly] = {}
        self.health_history: Dict[int, List[HealthMetrics]] = {}
        self.logger = logging.getLogger(__name__)
        self.anomaly_counter = 1
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.initialize_detection_models()
    
    def initialize_detection_models(self):
        """Initialize machine learning models for anomaly detection"""
        # Different models for different types of anomalies
        model_types = ['power', 'thermal', 'attitude', 'communication', 'propulsion']
        
        for model_type in model_types:
            self.models[model_type] = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.scalers[model_type] = StandardScaler()
    
    def add_health_metrics(self, metrics: HealthMetrics):
        """Add new health metrics for a satellite"""
        if metrics.satellite_id not in self.health_history:
            self.health_history[metrics.satellite_id] = []
        
        self.health_history[metrics.satellite_id].append(metrics)
        
        # Keep only last 1000 measurements
        if len(self.health_history[metrics.satellite_id]) > 1000:
            self.health_history[metrics.satellite_id] = self.health_history[metrics.satellite_id][-1000:]
        
        # Check for anomalies
        self.detect_anomalies(metrics.satellite_id)
    
    def detect_anomalies(self, satellite_id: int):
        """Detect anomalies in satellite health metrics"""
        if satellite_id not in self.health_history:
            return
        
        history = self.health_history[satellite_id]
        if len(history) < 10:  # Need minimum data points
            return
        
        latest_metrics = history[-1]
        
        # Check different subsystems
        self._check_power_anomalies(satellite_id, latest_metrics, history)
        self._check_thermal_anomalies(satellite_id, latest_metrics, history)
        self._check_attitude_anomalies(satellite_id, latest_metrics, history)
        self._check_communication_anomalies(satellite_id, latest_metrics, history)
        self._check_propulsion_anomalies(satellite_id, latest_metrics, history)
    
    def _check_power_anomalies(self, satellite_id: int, current: HealthMetrics, history: List[HealthMetrics]):
        """Check for power system anomalies"""
        # Extract power-related features
        features = []
        for h in history[-50:]:  # Last 50 measurements
            features.append([h.battery_voltage, h.solar_panel_current])
        
        if len(features) < 10:
            return
        
        features = np.array(features)
        
        # Train model if not enough data yet
        if not hasattr(self.models['power'], 'offset_'):
            scaled_features = self.scalers['power'].fit_transform(features)
            self.models['power'].fit(scaled_features)
            return
        
        # Check current measurement
        current_features = np.array([[current.battery_voltage, current.solar_panel_current]])
        scaled_current = self.scalers['power'].transform(current_features)
        
        anomaly_score = self.models['power'].decision_function(scaled_current)[0]
        is_anomaly = self.models['power'].predict(scaled_current)[0] == -1
        
        if is_anomaly:
            severity = self._calculate_severity(anomaly_score)
            self._create_anomaly(
                satellite_id=satellite_id,
                anomaly_type="power_system",
                severity=severity,
                description=f"Power system anomaly detected: Battery voltage {current.battery_voltage:.2f}V, Solar current {current.solar_panel_current:.2f}A",
                parameters=["battery_voltage", "solar_panel_current"],
                confidence=abs(anomaly_score)
            )
    
    def _check_thermal_anomalies(self, satellite_id: int, current: HealthMetrics, history: List[HealthMetrics]):
        """Check for thermal anomalies"""
        recent_temps = [h.temperature for h in history[-20:]]
        
        if len(recent_temps) < 5:
            return
        
        mean_temp = np.mean(recent_temps)
        std_temp = np.std(recent_temps)
        
        # Check for temperature spikes or drops
        if abs(current.temperature - mean_temp) > 3 * std_temp:
            severity = "high" if abs(current.temperature - mean_temp) > 5 * std_temp else "medium"
            self._create_anomaly(
                satellite_id=satellite_id,
                anomaly_type="thermal",
                severity=severity,
                description=f"Temperature anomaly: {current.temperature:.1f}K (mean: {mean_temp:.1f}K)",
                parameters=["temperature"],
                confidence=min(1.0, abs(current.temperature - mean_temp) / (5 * std_temp))
            )
    
    def _check_attitude_anomalies(self, satellite_id: int, current: HealthMetrics, history: List[HealthMetrics]):
        """Check for attitude control anomalies"""
        if current.attitude_error > 5.0:  # degrees
            severity = "critical" if current.attitude_error > 10.0 else "high"
            self._create_anomaly(
                satellite_id=satellite_id,
                anomaly_type="attitude_control",
                severity=severity,
                description=f"Attitude error exceeded threshold: {current.attitude_error:.2f}°",
                parameters=["attitude_error"],
                confidence=min(1.0, current.attitude_error / 15.0)
            )
    
    def _check_communication_anomalies(self, satellite_id: int, current: HealthMetrics, history: List[HealthMetrics]):
        """Check for communication system anomalies"""
        if current.communication_strength < 0.3:
            severity = "high" if current.communication_strength < 0.1 else "medium"
            self._create_anomaly(
                satellite_id=satellite_id,
                anomaly_type="communication",
                severity=severity,
                description=f"Low communication signal strength: {current.communication_strength:.2f}",
                parameters=["communication_strength"],
                confidence=1.0 - current.communication_strength
            )
    
    def _check_propulsion_anomalies(self, satellite_id: int, current: HealthMetrics, history: List[HealthMetrics]):
        """Check for propulsion system anomalies"""
        if current.thruster_pressure < 50.0:  # PSI
            severity = "high" if current.thruster_pressure < 20.0 else "medium"
            self._create_anomaly(
                satellite_id=satellite_id,
                anomaly_type="propulsion",
                severity=severity,
                description=f"Low thruster pressure: {current.thruster_pressure:.1f} PSI",
                parameters=["thruster_pressure"],
                confidence=1.0 - (current.thruster_pressure / 100.0)
            )
        
        if current.fuel_level < 10.0:  # Percentage
            severity = "critical" if current.fuel_level < 5.0 else "high"
            self._create_anomaly(
                satellite_id=satellite_id,
                anomaly_type="fuel_depletion",
                severity=severity,
                description=f"Low fuel level: {current.fuel_level:.1f}%",
                parameters=["fuel_level"],
                confidence=1.0 - (current.fuel_level / 20.0)
            )
    
    def _calculate_severity(self, anomaly_score: float) -> str:
        """Calculate severity based on anomaly score"""
        abs_score = abs(anomaly_score)
        if abs_score > 0.8:
            return "critical"
        elif abs_score > 0.6:
            return "high"
        elif abs_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _create_anomaly(self, satellite_id: int, anomaly_type: str, severity: str,
                       description: str, parameters: List[str], confidence: float):
        """Create a new anomaly record"""
        anomaly_id = self.anomaly_counter
        self.anomaly_counter += 1
        
        # Determine recommended action based on type and severity
        recommended_actions = {
            "power_system": "Check solar panel orientation and battery health",
            "thermal": "Investigate thermal control system",
            "attitude_control": "Perform attitude correction maneuver",
            "communication": "Check antenna alignment and transmitter power",
            "propulsion": "Inspect thruster system and fuel lines",
            "fuel_depletion": "Plan fuel conservation measures"
        }
        
        anomaly = Anomaly(
            id=anomaly_id,
            satellite_id=satellite_id,
            anomaly_type=anomaly_type,
            severity=severity,
            description=description,
            detected_at=datetime.now(),
            parameters_affected=parameters,
            confidence_score=min(1.0, confidence),
            recommended_action=recommended_actions.get(anomaly_type, "Investigate further"),
            status="new"
        )
        
        self.anomalies[anomaly_id] = anomaly
        self.logger.warning(f"Anomaly detected: {description}")
    
    def get_active_anomalies(self, satellite_id: Optional[int] = None) -> List[Dict]:
        """Get active anomalies"""
        active = []
        for anomaly in self.anomalies.values():
            if anomaly.status in ["new", "investigating"]:
                if satellite_id is None or anomaly.satellite_id == satellite_id:
                    active.append(asdict(anomaly))
        
        return sorted(active, key=lambda x: x["detected_at"], reverse=True)
    
    def get_anomaly_statistics(self) -> Dict:
        """Get anomaly statistics"""
        total_anomalies = len(self.anomalies)
        by_severity = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        by_type = {}
        resolved_count = 0
        
        for anomaly in self.anomalies.values():
            by_severity[anomaly.severity] += 1
            by_type[anomaly.anomaly_type] = by_type.get(anomaly.anomaly_type, 0) + 1
            if anomaly.status == "resolved":
                resolved_count += 1
        
        return {
            "total_anomalies": total_anomalies,
            "by_severity": by_severity,
            "by_type": by_type,
            "resolution_rate": (resolved_count / max(1, total_anomalies)) * 100,
            "active_anomalies": len(self.get_active_anomalies())
        }
    
    def resolve_anomaly(self, anomaly_id: int, resolution_notes: str = "") -> bool:
        """Mark an anomaly as resolved"""
        if anomaly_id not in self.anomalies:
            return False
        
        self.anomalies[anomaly_id].status = "resolved"
        self.logger.info(f"Anomaly {anomaly_id} resolved: {resolution_notes}")
        return True
    
    def generate_health_report(self, satellite_id: int) -> Dict:
        """Generate comprehensive health report for a satellite"""
        if satellite_id not in self.health_history:
            return {"error": "No health data available"}
        
        history = self.health_history[satellite_id]
        if not history:
            return {"error": "No health data available"}
        
        latest = history[-1]
        
        # Calculate trends
        if len(history) >= 10:
            recent_temps = [h.temperature for h in history[-10:]]
            temp_trend = "stable"
            if recent_temps[-1] > recent_temps[0] + 5:
                temp_trend = "increasing"
            elif recent_temps[-1] < recent_temps[0] - 5:
                temp_trend = "decreasing"
        else:
            temp_trend = "insufficient_data"
        
        # Overall health score
        health_score = self._calculate_health_score(latest)
        
        return {
            "satellite_id": satellite_id,
            "timestamp": latest.timestamp,
            "overall_health_score": health_score,
            "subsystem_status": {
                "power": "nominal" if latest.battery_voltage > 12.0 else "degraded",
                "thermal": "nominal" if 250 <= latest.temperature <= 350 else "warning",
                "attitude": "nominal" if latest.attitude_error < 2.0 else "warning",
                "communication": "nominal" if latest.communication_strength > 0.7 else "degraded",
                "propulsion": "nominal" if latest.fuel_level > 20.0 else "warning"
            },
            "trends": {
                "temperature": temp_trend,
                "fuel_consumption": "normal"  # Simplified
            },
            "active_anomalies": len(self.get_active_anomalies(satellite_id)),
            "recommendations": self._generate_recommendations(satellite_id, latest)
        }
    
    def _calculate_health_score(self, metrics: HealthMetrics) -> float:
        """Calculate overall health score (0-100)"""
        scores = []
        
        # Power system score
        power_score = min(100, (metrics.battery_voltage / 14.0) * 100)
        scores.append(power_score)
        
        # Thermal score
        temp_score = 100 if 250 <= metrics.temperature <= 350 else max(0, 100 - abs(metrics.temperature - 300) / 2)
        scores.append(temp_score)
        
        # Attitude score
        attitude_score = max(0, 100 - metrics.attitude_error * 10)
        scores.append(attitude_score)
        
        # Communication score
        comm_score = metrics.communication_strength * 100
        scores.append(comm_score)
        
        # Fuel score
        fuel_score = metrics.fuel_level
        scores.append(fuel_score)
        
        return np.mean(scores)
    
    def _generate_recommendations(self, satellite_id: int, metrics: HealthMetrics) -> List[str]:
        """Generate health recommendations"""
        recommendations = []
        
        if metrics.battery_voltage < 12.0:
            recommendations.append("Monitor battery health and charging system")
        
        if metrics.temperature > 350 or metrics.temperature < 250:
            recommendations.append("Check thermal control system")
        
        if metrics.attitude_error > 2.0:
            recommendations.append("Perform attitude calibration")
        
        if metrics.communication_strength < 0.5:
            recommendations.append("Check antenna alignment")
        
        if metrics.fuel_level < 15.0:
            recommendations.append("Plan fuel conservation measures")
        
        if not recommendations:
            recommendations.append("All systems nominal")
        
        return recommendations
