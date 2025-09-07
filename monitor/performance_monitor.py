import time

class PerformanceMonitor:
    def __init__(self):
        self.frame_times = []

    def start_frame(self):
        return time.perf_counter()

    def end_frame(self, start_time):
        frame_time = time.perf_counter() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)

    def get_metrics(self):
        if not self.frame_times:
            return {'fps': 0, 'avg_frame_time': 0, 'score': 0}
        avg = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg if avg > 0 else 0
        score = min(100, (fps / 60) * 100)
        return {'fps': round(fps, 1), 'avg_frame_time': round(avg * 1000, 1), 'score': round(score, 1)}
