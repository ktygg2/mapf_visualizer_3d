import rclpy
from rclpy.node import Node
import pandas as pd
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np

class CsvPathPublisher(Node):
    def __init__(self):
        super().__init__('csv_repulsive_visualizer')

        # CSV 파일 경로
        csv_path = 'path_smoothing_simple.csv'  # 필요 시 절대 경로로 수정

        # CSV 데이터 로드
        self.df = pd.read_csv(csv_path)
        self.df['agent'] = self.df['agent'].astype(int)
        self.time_steps = sorted(self.df['time'].unique())
        self.agents = sorted(self.df['agent'].unique())
        self.current_time_idx = 0

        # 각 에이전트의 Path 메시지 생성
        self.paths = {}
        for agent_id in self.agents:
            agent_df = self.df[self.df['agent'] == agent_id].sort_values('time')
            path_msg = Path()
            path_msg.header.frame_id = 'map'
            for _, row in agent_df.iterrows():
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.pose.position.x = float(row['x'])
                pose.pose.position.y = float(row['y'])
                pose.pose.position.z = float(row['z'])
                path_msg.poses.append(pose)
            self.paths[agent_id] = path_msg

        # 통합된 경로 마커 발행자
        self.path_marker_pub = self.create_publisher(MarkerArray, '/all_agents_paths', 10)
        # 움직이는 에이전트 마커 발행자
        self.marker_pub = self.create_publisher(MarkerArray, '/moving_agents', 10)

        # 주기적 타이머 실행
        self.timer = self.create_timer(0.1, self.timer_callback)

    def get_agent_color(self, agent_id):
        """에이전트 고유 색상 생성"""
        np.random.seed(agent_id)
        color = ColorRGBA()
        color.r = np.random.uniform(0.2, 1.0)
        color.g = np.random.uniform(0.2, 1.0)
        color.b = np.random.uniform(0.2, 1.0)
        color.a = 1.0
        return color

    def timer_callback(self):
        # === 경로 전체 마커 생성 ===
        path_marker_array = MarkerArray()

        for agent_id, path_msg in self.paths.items():
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"path_agent_{agent_id}"
            marker.id = int(agent_id)
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05  # 선 두께
            marker.color = self.get_agent_color(agent_id)
            marker.pose.orientation.w = 1.0

            for pose in path_msg.poses:
                point = Point()
                point.x = pose.pose.position.x
                point.y = pose.pose.position.y
                point.z = pose.pose.position.z
                marker.points.append(point)

            path_marker_array.markers.append(marker)

        # 발행
        self.path_marker_pub.publish(path_marker_array)

        # === 현재 시간의 에이전트 위치 마커 생성 ===
        marker_array = MarkerArray()

        # 기존 마커 삭제 명령
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # 현재 시간 추출
        current_time = self.time_steps[self.current_time_idx]

        for agent_id in self.agents:
            agent_df = self.df[(self.df['agent'] == agent_id) & (self.df['time'] == current_time)]
            if not agent_df.empty:
                x = float(agent_df.iloc[0]['x'])
                y = float(agent_df.iloc[0]['y'])
                z = float(agent_df.iloc[0]['z'])

                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"agent_{agent_id}"
                marker.id = 0
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

        # 다음 시간으로 이동 (순환)
        self.current_time_idx += 1
        if self.current_time_idx >= len(self.time_steps):
            self.current_time_idx = 0

def main(args=None):
    rclpy.init(args=args)
    node = CsvPathPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
