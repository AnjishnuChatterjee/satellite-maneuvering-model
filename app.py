# Update the display_collision_analysis method in app.py
def display_collision_analysis(self):
    st.subheader("Collision Risk Assessment")
    
    # Get real collision predictions from debris tracker
    satellites = self.satellite_api.get_all_satellites()
    satellite_positions = [{
        'id': sat.id,
        'position': sat.position,
        'velocity': sat.velocity
    } for sat in satellites]
    
    collision_predictions = self.debris_tracker.predict_collisions(satellite_positions)
    
    if collision_predictions:
        collision_data = []
        for pred in collision_predictions[:10]:  # Show top 10
            collision_data.append({
                'Satellite': f"SAT-{pred['satellite_id']}",
                'Debris': pred['debris_name'],
                'Distance (km)': round(pred['minimum_distance_km'], 2),
                'Probability': pred['collision_probability'],
                'TCA (hours)': round(pred['time_to_closest_approach'], 2),
                'Status': pred['threat_level']
            })
        
        st.dataframe(pd.DataFrame(collision_data), use_container_width=True)
    else:
        st.info("No collision threats detected in the next 24 hours.")
    
    # Risk visualization (only if we have collision data)
    if collision_predictions:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(self.create_risk_chart(pd.DataFrame(collision_data)), use_container_width=True)
        with col2:
            st.plotly_chart(self.create_timeline_chart(pd.DataFrame(collision_data)), use_container_width=True)