# Update plotly_renderer.py to visualize IR detection and maneuvering

import plotly.graph_objects as go
import numpy as np
from utils.simulation_math import simulate_ir_detection, DANGER_THRESHOLD

class PlotlyRenderer:
    def __init__(self, config):
        self.config = config
        self.satellite_positions = []
        self.debris_positions = []
        self.detection_data = []
        self.maneuver_data = []

    def generate_figure(self):
        # Generate sample data for demonstration
        self._generate_sample_data()
        
        fig = go.Figure()
        
        # Add Earth
        fig.add_trace(go.Surface(
            x=self.earth_x, y=self.earth_y, z=self.earth_z,
            colorscale=[[0, 'rgb(30, 144, 255)'], [1, 'rgb(30, 144, 255)']],
            showscale=False,
            opacity=0.7
        ))
        
        # Add satellites
        sat_x, sat_y, sat_z = zip(*self.satellite_positions)
        fig.add_trace(go.Scatter3d(
            x=sat_x, y=sat_y, z=sat_z,
            mode='markers',
            marker=dict(size=8, color='cyan'),
            name='Satellites'
        ))
        
        # Add debris
        debris_x, debris_y, debris_z = zip(*self.debris_positions)
        fig.add_trace(go.Scatter3d(
            x=debris_x, y=debris_y, z=debris_z,
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Debris'
        ))
        
        # Add detection ranges (spheres around satellites)
        for i, sat_pos in enumerate(self.satellite_positions):
            detection_range = self._create_detection_sphere(sat_pos, DANGER_THRESHOLD)
            fig.add_trace(go.Surface(
                x=detection_range[0], y=detection_range[1], z=detection_range[2],
                colorscale=[[0, 'rgba(255, 0, 0, 0.1)'], [1, 'rgba(255, 0, 0, 0.1)']],
                showscale=False,
                opacity=0.2,
                name=f'Sat {i+1} Detection Range'
            ))
        
        # Add detection lines for detected debris
        for detection in self.detection_data:
            sat_idx, debris_idx, detection_strength = detection
            if detection_strength > 0.3:  # Only show strong detections
                sat_pos = self.satellite_positions[sat_idx]
                debris_pos = self.debris_positions[debris_idx]
                
                fig.add_trace(go.Scatter3d(
                    x=[sat_pos[0], debris_pos[0]],
                    y=[sat_pos[1], debris_pos[1]],
                    z=[sat_pos[2], debris_pos[2]],
                    mode='lines',
                    line=dict(color='yellow', width=2, dash='dash'),
                    name=f'Detection {sat_idx}-{debris_idx}'
                ))
        
        # Add maneuver vectors
        for maneuver in self.maneuver_data:
            sat_idx, maneuver_vector = maneuver
            sat_pos = self.satellite_positions[sat_idx]
            end_pos = [sat_pos[i] + maneuver_vector[i] * 500 for i in range(3)]  # Scale for visibility
            
            fig.add_trace(go.Scatter3d(
                x=[sat_pos[0], end_pos[0]],
                y=[sat_pos[1], end_pos[1]],
                z=[sat_pos[2], end_pos[2]],
                mode='lines',
                line=dict(color='orange', width=4),
                name=f'Maneuver {sat_idx}'
            ))
        
        fig.update_layout(
            title="Orbital Simulation with IR Detection and Maneuvering",
            scene=dict(
                bgcolor='black',
                xaxis=dict(title='X (km)', backgroundcolor='black'),
                yaxis=dict(title='Y (km)', backgroundcolor='black'),
                zaxis=dict(title='Z (km)', backgroundcolor='black')
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def _generate_sample_data(self):
        # Generate sample Earth
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = 6371 * np.outer(np.cos(u), np.sin(v))
        y = 6371 * np.outer(np.sin(u), np.sin(v))
        z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
        self.earth_x, self.earth_y, self.earth_z = x, y, z
        
        # Generate sample satellite positions
        self.satellite_positions = [
            [8000, 0, 0],
            [0, 8000, 0],
            [0, 0, 8000]
        ]
        
        # Generate sample debris positions
        self.debris_positions = [
            [7500, 500, 100],
            [500, 7500, 200],
            [300, 300, 7800],
            [7200, 800, 300],
            [600, 7200, 400]
        ]
        
        # Simulate IR detection
        debris_temperatures = [0.8, 0.6, 0.9, 0.7, 0.5]  # Simulated temperatures
        self.detection_data = []
        
        for i, sat_pos in enumerate(self.satellite_positions):
            detection_probs = simulate_ir_detection(sat_pos, self.debris_positions, debris_temperatures)
            for j, prob in enumerate(detection_probs):
                self.detection_data.append((i, j, prob))
        
        # Simulate maneuvers
        self.maneuver_data = [
            (0, [100, 50, 0]),  # Satellite 0 maneuver
            (1, [0, 0, 100])    # Satellite 1 maneuver
        ]
    
    def _create_detection_sphere(self, center, radius, resolution=20):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        return x, y, z