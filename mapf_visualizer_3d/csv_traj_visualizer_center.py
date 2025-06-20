import rclpy
from rclpy.node import Node
import pandas as pd
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np

class ReplanningVisualizer(Node):
    def __init__(self):
        super().__init__('csv_traj_visualizer_center')
        
        # CSV 파일 경로 설정
        csv_path = 'neighbor_center_smoothing_all.csv'  # 수정된 CSV 파일
        self.df = pd.read_csv(csv_path)
        
        # 데이터 전처리
        self.df['agent'] = self.df['agent'].astype(int)
        if 'replan_iter' not in self.df.columns:
            self.df['replan_iter'] = 1  # 반복 주기 정보가 없는 경우 기본값
        self.max_iter = int(self.df['replan_iter'].max())
        self.agents = sorted(self.df['agent'].unique())
        self.time_steps = sorted(self.df['time'].unique())
        
        # 시각화 파라미터
        self.line_width = 0.08  # 경로 선 두께
        self.agent_radius = 0.3  # 에이전트 반경
        
        # 발행자 설정
        self.marker_pub = self.create_publisher(MarkerArray, '/replanning_paths', 10)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.current_time_idx = 0

    def get_iteration_color(self, agent_id, iteration):
        """반복 주기별 색상 및 투명도 설정"""
        base_hue = agent_id / len(self.agents)  # 에이전트별 고유 hue 값
        return ColorRGBA(
            r=base_hue,
            g=(base_hue + 0.5) % 1.0,
            b=(base_hue + 0.3) % 1.0,
            a=0.2 + 0.8 * (iteration/self.max_iter)  # 최신 반복일수록 불투명
        )

    def timer_callback(self):
        marker_array = MarkerArray()
        
        # 모든 반복 주기 처리
        for iteration in range(1, self.max_iter+1):
            for agent_id in self.agents:
                # 경로 데이터 필터링
                agent_df = self.df[
                    (self.df['agent'] == agent_id) &
                    (self.df['replan_iter'] == iteration) &
                    (self.df['time'] <= self.time_steps[self.current_time_idx])
                ].sort_values('time')
                
                if len(agent_df) >= 2:
                    # 라인 마커 생성
                    line_marker = Marker()
                    line_marker.header.frame_id = "map"
                    line_marker.header.stamp = self.get_clock().now().to_msg()
                    line_marker.ns = f"agent_{agent_id}_iter_{iteration}"
                    line_marker.id = int(agent_id * 1000 + iteration)  # 고유 ID 생성
                    line_marker.type = Marker.LINE_STRIP
                    line_marker.action = Marker.ADD
                    line_marker.scale.x = self.line_width
                    line_marker.color = self.get_iteration_color(agent_id, iteration)
                    
                    # 경로 포인트 추가
                    for _, row in agent_df.iterrows():
                        point = Point()
                        point.x = float(row['x'])
                        point.y = float(row['y'])
                        point.z = float(row['z'])
                        line_marker.points.append(point)
                    marker_array.markers.append(line_marker)

                    # 현재 위치 마커 (최신 반복만 표시)
                    if iteration == self.max_iter:
                        current_pos = agent_df.iloc[-1]
                        sphere_marker = Marker()
                        sphere_marker.header.frame_id = "map"
                        sphere_marker.header.stamp = self.get_clock().now().to_msg()
                        sphere_marker.ns = f"agent_{agent_id}_current"
                        sphere_marker.id = int(agent_id * 1000 + 999)
                        sphere_marker.type = Marker.SPHERE
                        sphere_marker.action = Marker.ADD
                        sphere_marker.pose.position.x = current_pos['x']
                        sphere_marker.pose.position.y = current_pos['y']
                        sphere_marker.pose.position.z = current_pos['z']
                        sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = self.agent_radius
                        sphere_marker.color = self.get_iteration_color(agent_id, iteration)
                        sphere_marker.color.a = 1.0  # 현재 위치는 불투명
                        marker_array.markers.append(sphere_marker)

        # 마커 발행
        self.marker_pub.publish(marker_array)
        
        # 시간 인덱스 업데이트 (순환)
        self.current_time_idx += 1
        if self.current_time_idx >= len(self.time_steps):
            self.current_time_idx = 0
            self.get_logger().info('경로 애니메이션 재시작')

def main(args=None):
    rclpy.init(args=args)
    node = ReplanningVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
