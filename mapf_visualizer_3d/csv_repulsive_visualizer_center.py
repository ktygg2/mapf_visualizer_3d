import rclpy
from rclpy.node import Node
import pandas as pd
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np

class CenterBasedVisualizer(Node):
    def __init__(self):
        super().__init__('csv_repulsive_visualizer_center')

        # CSV 파일 경로
        csv_path = 'neighbor_center_smoothing_simple.csv'

        # 데이터 로드
        self.df = pd.read_csv(csv_path)
        self.df['agent'] = self.df['agent'].astype(int)
        self.time_steps = sorted(self.df['time'].unique())
        self.agents = sorted(self.df['agent'].unique())
        self.current_time_idx = 0

        # 전체 경로를 마커로 표시
        self.path_marker_pub = self.create_publisher(MarkerArray, '/center_agent_paths', 10)

        # 움직이는 에이전트 마커 발행자
        self.marker_pub = self.create_publisher(MarkerArray, '/center_moving_agents', 10)

        # 마커 메시지 생성
        self.path_marker_array = self.create_path_markers()
        self.path_marker_pub.publish(self.path_marker_array)  # 처음 한 번만 발행

        # 타이머로 움직이는 마커만 주기적 발행
        self.timer = self.create_timer(0.1, self.timer_callback)

    def get_agent_color(self, agent_id):
        np.random.seed(agent_id)
        color = ColorRGBA()
        color.r = np.random.uniform(0.2, 1.0)
        color.g = np.random.uniform(0.2, 1.0)
        color.b = np.random.uniform(0.2, 1.0)
        color.a = 1.0
        return color

    def create_path_markers(self):
        marker_array = MarkerArray()
        for idx, agent_id in enumerate(self.agents):
            agent_df = self.df[self.df['agent'] == agent_id].sort_values('time')

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"path_agent_{agent_id}"
            marker.id = int(agent_id)
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05  # 선의 굵기
            marker.color = self.get_agent_color(agent_id)

            for _, row in agent_df.iterrows():
                point = Point()
                point.x = float(row['x'])
                point.y = float(row['y'])
                point.z = float(row['z'])
                marker.points.append(point)

            marker_array.markers.append(marker)
        return marker_array

    def timer_callback(self):
        # 1. 전체 경로 마커 반복 발행
        for marker in self.path_marker_array.markers:
            marker.header.stamp = self.get_clock().now().to_msg()
        self.path_marker_pub.publish(self.path_marker_array)

        # 2. 움직이는 마커들
        marker_array = MarkerArray()

        # 기존 마커 제거
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        current_time = self.time_steps[self.current_time_idx]

        for agent_id in self.agents:
            agent_df = self.df[(self.df['agent'] == agent_id) &
                            (self.df['time'] == current_time)]
            if not agent_df.empty:
                x = float(agent_df.iloc[0]['x'])
                y = float(agent_df.iloc[0]['y'])
                z = float(agent_df.iloc[0]['z'])

                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"center_agent_{agent_id}"
                marker.id = int(agent_id)
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = z
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 0.3
                marker.color = self.get_agent_color(agent_id)
                marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

        self.current_time_idx += 1
        if self.current_time_idx >= len(self.time_steps):
            self.current_time_idx = 0
            
def main(args=None):
    rclpy.init(args=args)
    node = CenterBasedVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
