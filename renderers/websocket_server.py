# Update the run_simulation method to handle data format conversion:
async def run_simulation(self):
    start = time.time()
    self.simulation_running = True
    
    while self.simulation_running:
        t = time.time() - start
        self.frame_count += 1
        
        # Update positions
        self.update_positions(t)
        
        # Simulate IR detection and maneuvering
        self.simulate_detection_and_maneuvering()
        
        # Prepare data for clients - ensure all values are JSON serializable
        satellites_data = []
        for sat in self.satellites:
            satellites_data.append({
                'id': sat['id'],
                'position': [float(x) for x in sat['position']],
                'threat_level': float(sat['threat_level']),
                'fuel': float(sat['fuel']),
                'maneuvering': sat['maneuvering']
            })
        
        debris_data = []
        for d in self.debris:
            debris_data.append({
                'id': d['id'],
                'position': [float(x) for x in d['position']],
                'type': 'fragment',
                'temperature': float(d['temperature'])
            })
        
        # Send update to clients
        await self.broadcast({
            'type': 'position_update', 
            'timestamp': float(t), 
            'satellites': satellites_data, 
            'debris': debris_data,
            'detections': self.get_detection_data(),
            'maneuvers': self.get_maneuver_data()
        })
        
        await asyncio.sleep(1 / 30)