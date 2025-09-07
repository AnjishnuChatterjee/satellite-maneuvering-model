def create_ghosting_traces(positions, num_ghosts=5):
    import plotly.graph_objects as go
    traces = []
    for i in range(num_ghosts):
        alpha = 1.0 - (i * 0.15)
        if len(positions) > i:
            pos = positions[-(i+1)]
            traces.append(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers',
                marker=dict(size=12 - i*2, color=f'rgba(0,255,255,{alpha})'),
                showlegend=False
            ))
    return traces
