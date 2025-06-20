import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import csv
from collections import defaultdict
import ast

class CsvVisualizer(Node):
    def __init__(self):
        super().__init__('csv_visualizer')
        self.publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 100)
        self.timer = self.create_timer(0.2, self.timer_callback)

        self.paths = self.load_paths('/home/kty/ros2_ws/src/mapf_visualizer_3d/scripts/result.csv')
        self.starts, self.goals = self.load_start_goal('/home/kty/ros2_ws/src/mapf_visualizer_3d/scripts/instance_config.csv')

        self.current_time = 0.0
        self.max_time = max(max(t.keys()) for t in self.paths.values())
        self.traces = defaultdict(list)

        self.interpolated_paths = {
            agent: self.interpolate_path(path, steps_per_segment=5)
            for agent, path in self.paths.items()
        }

        self.publish_start_goal_markers()
        self.save_interpolated_paths_to_csv('/home/kty/ros2_ws/src/mapf_visualizer_3d/scripts/result_interpolator.csv')

    def save_interpolated_paths_to_csv(self, output_path):
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['agent', 'time', 'x', 'y', 'z']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for agent_id, path in self.interpolated_paths.items():
                for t, (x, y, z) in sorted(path.items()):
                    writer.writerow({
                        'agent': agent_id,
                        'time': round(t, 2),
                        'x': x,
                        'y': y,
                        'z': z
                    })

    def load_paths(self, filename):
        paths = defaultdict(dict)
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                agent = int(row['agent'])
                time = int(row['time'])
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                paths[agent][time] = (x, y, z)
        return paths

    def load_start_goal(self, result_txt_path):
        starts, goals = {}, {}
        with open(result_txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("starts="):
                    start_list = ast.literal_eval("[" + line.split("=")[1].strip() + "]")
                    for i, (x, y, z) in enumerate(start_list):
                        starts[i] = (x, y, z)
                elif line.startswith("goals="):
                    goal_list = ast.literal_eval("[" + line.split("=")[1].strip() + "]")
                    for i, (x, y, z) in enumerate(goal_list):
                        goals[i] = (x, y, z)
        return starts, goals

    def interpolate_path(self, agent_path, steps_per_segment=5):
        interpolated = {}
        times = sorted(agent_path.keys())
        for i in range(len(times) - 1):
            t0, t1 = times[i], times[i + 1]
            p0 = agent_path[t0]
            p1 = agent_path[t1]
            for step in range(steps_per_segment):
                ratio = step / steps_per_segment
                x = p0[0] + (p1[0] - p0[0]) * ratio
                y = p0[1] + (p1[1] - p0[1]) * ratio
                z = p0[2] + (p1[2] - p0[2]) * ratio
                interpolated_time = t0 + ratio
                interpolated[interpolated_time] = (x, y, z)
        interpolated[times[-1]] = agent_path[times[-1]]
        return interpolated

    def publish_start_goal_markers(self):
        marker_array = MarkerArray()
        for agent_id in self.starts:
            for kind, position in zip(['start', 'goal'], [self.starts[agent_id], self.goals[agent_id]]):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"{kind}_{agent_id}"
                marker.id = 100000 + agent_id if kind == 'start' else 200000 + agent_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = position[0]
                marker.pose.position.y = position[1]
                marker.pose.position.z = position[2]
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 0.3
                if kind == 'start':
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                else:
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                marker.color.a = 1.0
                marker_array.markers.append(marker)
        self.publisher.publish(marker_array)

    def timer_callback(self):
        if self.current_time > self.max_time:
            self.get_logger().info('시각화 완료!')
            return

        marker_array = MarkerArray()

        for agent_id, path in self.paths.items():
            if int(self.current_time) in path:
                x, y, z = path[int(self.current_time)]
                self.traces[agent_id].append(Point(x=x, y=y, z=z))

                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"agent_{agent_id}"
                marker.id = agent_id * 100000 + int(self.current_time * 100)
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = z
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.r = (agent_id * 37 % 255) / 255.0
                marker.color.g = (agent_id * 73 % 255) / 255.0
                marker.color.b = (agent_id * 113 % 255) / 255.0
                marker.color.a = 1.0
                marker_array.markers.append(marker)

                trace_marker = Marker()
                trace_marker.header.frame_id = "map"
                trace_marker.header.stamp = self.get_clock().now().to_msg()
                trace_marker.ns = f"trace_{agent_id}"
                trace_marker.id = 10000 + agent_id * 1000 + int(self.current_time)
                trace_marker.type = Marker.SPHERE
                trace_marker.action = Marker.ADD
                trace_marker.pose.position.x = x
                trace_marker.pose.position.y = y
                trace_marker.pose.position.z = z
                trace_marker.pose.orientation.w = 1.0
                trace_marker.scale.x = 0.1
                trace_marker.scale.y = 0.1
                trace_marker.scale.z = 0.1
                trace_marker.color.r = (agent_id * 37 % 255) / 255.0
                trace_marker.color.g = (agent_id * 73 % 255) / 255.0
                trace_marker.color.b = (agent_id * 113 % 255) / 255.0
                trace_marker.color.a = 1.0
                marker_array.markers.append(trace_marker)
            
            # 경로 선(Line Strip) 시각화
            line_marker = Marker()
            line_marker.header.frame_id = "map"
            line_marker.header.stamp = self.get_clock().now().to_msg()
            line_marker.ns = f"path_{agent_id}"
            line_marker.id = 1000 + agent_id
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.05
            line_marker.color.r = (agent_id * 37 % 255) / 255.0
            line_marker.color.g = (agent_id * 73 % 255) / 255.0
            line_marker.color.b = (agent_id * 113 % 255) / 255.0
            line_marker.color.a = 0.5
            line_marker.points = self.traces[agent_id]
            marker_array.markers.append(line_marker)
        
        # 선형 보간된 점 시각화
        for agent_id, path in self.interpolated_paths.items():
            for t, (x, y, z) in path.items():
                if abs(t - self.current_time) < 0.01:
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = f"interpolated_{agent_id}"
                    marker.id = 50000 + agent_id * 1000 + int(t * 100)
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.pose.position.x = x
                    marker.pose.position.y = y
                    marker.pose.position.z = z
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.15
                    marker.scale.y = 0.15
                    marker.scale.z = 0.15
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 0.6
                    marker_array.markers.append(marker)

        self.publisher.publish(marker_array)
        self.current_time += 0.2

def main(args=None):
    rclpy.init(args=args)
    node = CsvVisualizer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
