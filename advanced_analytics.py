"""
Advanced Analytics Module for orbital dashboard
Provides predictive analytics, trend analysis, and machine learning capabilities
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import logging

class PredictiveAnalytics:
    """Advanced predictive analytics for satellite operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.historical_data = {}
        
    def add_historical_data(self, satellite_id: int, timestamp: datetime, 
                          metrics: Dict[str, float]):
        """Add historical data point for analysis"""
        if satellite_id not in self.historical_data:
            self.historical_data[satellite_id] = []
        
        data_point = {'timestamp': timestamp, **metrics}
        self.historical_data[satellite_id].append(data_point)
        
        # Keep only last 1000 data points
        if len(self.historical_data[satellite_id]) > 1000:
            self.historical_data[satellite_id] = self.historical_data[satellite_id][-1000:]
    
    def predict_fuel_depletion(self, satellite_id: int) -> Dict:
        """Predict when satellite will run out of fuel"""
        if satellite_id not in self.historical_data:
            return {"error": "No historical data available"}
        
        data = self.historical_data[satellite_id]
        if len(data) < 10:
            return {"error": "Insufficient data for prediction"}
        
        # Extract fuel levels and timestamps
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        if 'fuel_level' not in df.columns:
            return {"error": "No fuel data available"}
        
        # Create time-based features
        df['days_since_start'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 86400
        
        # Fit linear regression to fuel consumption
        X = df[['days_since_start']].values
        y = df['fuel_level'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict when fuel reaches 0
        current_fuel = y[-1]
        fuel_rate = model.coef_[0]  # fuel per day
        
        if fuel_rate >= 0:
            return {"prediction": "Fuel consumption rate is positive or zero"}
        
        days_remaining = -current_fuel / fuel_rate
        depletion_date = datetime.now() + timedelta(days=days_remaining)
        
        return {
            "days_remaining": max(0, days_remaining),
            "depletion_date": depletion_date,
            "current_fuel": current_fuel,
            "consumption_rate_per_day": -fuel_rate,
            "confidence": model.score(X, y)
        }
    
    def predict_collision_probability(self, satellite_positions: List[Dict], 
                                    debris_positions: List[Dict]) -> List[Dict]:
        """Predict collision probabilities using machine learning"""
        predictions = []
        
        for sat in satellite_positions:
            for debris in debris_positions:
                # Calculate relative position and velocity
                rel_pos = np.array(debris['position']) - np.array(sat['position'])
                rel_vel = np.array(debris['velocity']) - np.array(sat['velocity'])
                
                distance = np.linalg.norm(rel_pos)
                relative_speed = np.linalg.norm(rel_vel)
                
                # Simple probability model based on distance and relative speed
                base_prob = max(0, 1 - distance / 1000)  # Closer = higher probability
                speed_factor = min(1, relative_speed / 10)  # Faster = higher probability
                
                collision_prob = base_prob * speed_factor * 0.01  # Scale down
                
                if collision_prob > 0.001:  # Only report significant probabilities
                    predictions.append({
                        'satellite_id': sat['id'],
                        'debris_id': debris['id'],
                        'collision_probability': collision_prob,
                        'distance_km': distance,
                        'relative_speed_kms': relative_speed,
                        'time_to_closest_approach': self._calculate_tca(rel_pos, rel_vel)
                    })
        
        return sorted(predictions, key=lambda x: x['collision_probability'], reverse=True)
    
    def _calculate_tca(self, rel_pos: np.ndarray, rel_vel: np.ndarray) -> float:
        """Calculate time to closest approach"""
        if np.linalg.norm(rel_vel) < 1e-6:
            return float('inf')
        
        tca = -np.dot(rel_pos, rel_vel) / np.dot(rel_vel, rel_vel)
        return max(0, tca / 3600)  # Convert to hours
    
    def analyze_orbital_decay(self, satellite_id: int) -> Dict:
        """Analyze orbital decay trends"""
        if satellite_id not in self.historical_data:
            return {"error": "No historical data available"}
        
        data = self.historical_data[satellite_id]
        if len(data) < 20:
            return {"error": "Insufficient data for decay analysis"}
        
        df = pd.DataFrame(data)
        if 'altitude' not in df.columns:
            return {"error": "No altitude data available"}
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate decay rate
        time_diff = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400
        altitude_change = df['altitude'].iloc[-1] - df['altitude'].iloc[0]
        decay_rate = altitude_change / time_diff  # km per day
        
        # Predict reentry time (simplified)
        current_altitude = df['altitude'].iloc[-1]
        if decay_rate >= 0:
            reentry_days = float('inf')
        else:
            # Assume reentry at 100km altitude
            reentry_days = (current_altitude - 100) / (-decay_rate)
        
        return {
            "current_altitude_km": current_altitude,
            "decay_rate_km_per_day": decay_rate,
            "days_to_reentry": reentry_days if reentry_days != float('inf') else None,
            "reentry_date": datetime.now() + timedelta(days=reentry_days) if reentry_days != float('inf') else None,
            "trend": "decaying" if decay_rate < -0.1 else "stable" if abs(decay_rate) <= 0.1 else "boosting"
        }
    
    def detect_anomalous_patterns(self, satellite_id: int) -> List[Dict]:
        """Detect anomalous patterns in satellite behavior"""
        if satellite_id not in self.historical_data:
            return []
        
        data = self.historical_data[satellite_id]
        if len(data) < 50:
            return []
        
        df = pd.DataFrame(data)
        anomalies = []
        
        # Check for temperature anomalies
        if 'temperature' in df.columns:
            temp_mean = df['temperature'].mean()
            temp_std = df['temperature'].std()
            
            for i, row in df.iterrows():
                if abs(row['temperature'] - temp_mean) > 3 * temp_std:
                    anomalies.append({
                        'type': 'temperature_anomaly',
                        'timestamp': row['timestamp'],
                        'value': row['temperature'],
                        'expected_range': f"{temp_mean - 2*temp_std:.1f} - {temp_mean + 2*temp_std:.1f}",
                        'severity': 'high' if abs(row['temperature'] - temp_mean) > 4 * temp_std else 'medium'
                    })
        
        # Check for power anomalies
        if 'battery_voltage' in df.columns:
            voltage_trend = df['battery_voltage'].rolling(window=10).mean()
            recent_trend = voltage_trend.iloc[-10:].mean()
            historical_trend = voltage_trend.iloc[:-10].mean()
            
            if recent_trend < historical_trend * 0.9:
                anomalies.append({
                    'type': 'power_degradation',
                    'timestamp': df['timestamp'].iloc[-1],
                    'current_voltage': recent_trend,
                    'historical_voltage': historical_trend,
                    'degradation_percent': ((historical_trend - recent_trend) / historical_trend) * 100,
                    'severity': 'high'
                })
        
        return anomalies
    
    def generate_performance_forecast(self, satellite_id: int, days_ahead: int = 30) -> Dict:
        """Generate performance forecast for satellite"""
        if satellite_id not in self.historical_data:
            return {"error": "No historical data available"}
        
        data = self.historical_data[satellite_id]
        if len(data) < 30:
            return {"error": "Insufficient data for forecasting"}
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        forecast = {}
        
        # Forecast fuel consumption
        if 'fuel_level' in df.columns:
            fuel_prediction = self.predict_fuel_depletion(satellite_id)
            if 'days_remaining' in fuel_prediction:
                forecast['fuel'] = {
                    'current_level': df['fuel_level'].iloc[-1],
                    'predicted_level_30_days': max(0, df['fuel_level'].iloc[-1] - 
                                                 fuel_prediction['consumption_rate_per_day'] * days_ahead),
                    'depletion_risk': 'high' if fuel_prediction['days_remaining'] < days_ahead else 'low'
                }
        
        # Forecast orbital decay
        decay_analysis = self.analyze_orbital_decay(satellite_id)
        if 'decay_rate_km_per_day' in decay_analysis:
            current_alt = decay_analysis['current_altitude_km']
            predicted_alt = current_alt + decay_analysis['decay_rate_km_per_day'] * days_ahead
            
            forecast['orbital'] = {
                'current_altitude_km': current_alt,
                'predicted_altitude_km': predicted_alt,
                'altitude_change_km': predicted_alt - current_alt,
                'reentry_risk': 'high' if predicted_alt < 200 else 'low'
            }
        
        # Overall health forecast
        anomalies = self.detect_anomalous_patterns(satellite_id)
        forecast['health'] = {
            'current_anomalies': len(anomalies),
            'risk_level': 'high' if len(anomalies) > 3 else 'medium' if len(anomalies) > 1 else 'low',
            'recommended_actions': self._generate_forecast_recommendations(forecast)
        }
        
        return forecast
    
    def _generate_forecast_recommendations(self, forecast: Dict) -> List[str]:
        """Generate recommendations based on forecast"""
        recommendations = []
        
        if 'fuel' in forecast and forecast['fuel']['depletion_risk'] == 'high':
            recommendations.append("Plan fuel conservation measures")
        
        if 'orbital' in forecast and forecast['orbital']['reentry_risk'] == 'high':
            recommendations.append("Consider orbit boost maneuver")
        
        if 'health' in forecast and forecast['health']['risk_level'] == 'high':
            recommendations.append("Schedule comprehensive health check")
        
        if not recommendations:
            recommendations.append("Continue normal operations")
        
        return recommendations
    
    def calculate_mission_success_probability(self, mission_params: Dict) -> float:
        """Calculate probability of mission success based on parameters"""
        base_probability = 0.95
        
        # Adjust based on fuel availability
        fuel_factor = min(1.0, mission_params.get('fuel_available', 100) / 
                         mission_params.get('fuel_required', 10))
        
        # Adjust based on satellite health
        health_factor = mission_params.get('health_score', 100) / 100
        
        # Adjust based on mission complexity
        complexity_factor = 1.0 - (mission_params.get('complexity', 1) - 1) * 0.1
        
        # Adjust based on environmental conditions
        environment_factor = 1.0 - mission_params.get('debris_density', 0) * 0.1
        
        success_probability = (base_probability * fuel_factor * health_factor * 
                             complexity_factor * environment_factor)
        
        return max(0.0, min(1.0, success_probability))
    
    def optimize_constellation_coverage(self, satellites: List[Dict], 
                                      target_regions: List[Dict]) -> Dict:
        """Optimize satellite constellation for maximum coverage"""
        coverage_analysis = {
            'total_satellites': len(satellites),
            'target_regions': len(target_regions),
            'coverage_gaps': [],
            'optimization_suggestions': []
        }
        
        # Simplified coverage analysis
        covered_regions = 0
        for region in target_regions:
            region_covered = False
            for sat in satellites:
                # Simple distance-based coverage check
                distance = np.linalg.norm(
                    np.array(sat['position'][:2]) - np.array(region['coordinates'])
                )
                if distance < sat.get('coverage_radius', 1000):  # km
                    region_covered = True
                    break
            
            if region_covered:
                covered_regions += 1
            else:
                coverage_analysis['coverage_gaps'].append(region)
        
        coverage_percentage = (covered_regions / len(target_regions)) * 100
        coverage_analysis['coverage_percentage'] = coverage_percentage
        
        # Generate optimization suggestions
        if coverage_percentage < 90:
            coverage_analysis['optimization_suggestions'].append(
                "Consider repositioning satellites to improve coverage"
            )
        
        if len(coverage_analysis['coverage_gaps']) > 0:
            coverage_analysis['optimization_suggestions'].append(
                f"Deploy additional satellites to cover {len(coverage_analysis['coverage_gaps'])} gap regions"
            )
        
        return coverage_analysis
