import rclpy
from rclpy.node import Node
import pandas as pd
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np

class AllIterationsVisualizer(Node):
    def __init__(self):
        super().__init__('all_iterations_visualizer')
        
        # CSV 파일 경로
        csv_path = 'path_smoothing_all.csv'  # 수정된 파일명
        self.df = pd.read_csv(csv_path)
        self.df['agent'] = self.df['agent'].astype(int)
        self.df['replan_iter'] = self.df['replan_iter'].astype(int)

        # 데이터 정보
        self.max_iter = self.df['replan_iter'].max()
        self.agents = sorted(self.df['agent'].unique())
        self.time_steps = sorted(self.df['time'].unique())
                
        # 시각화 파라미터
        self.current_time_idx = 0
        self.line_width = 0.05  # 경로 선 두께
        
        # 발행자 및 타이머
        self.marker_pub = self.create_publisher(MarkerArray, '/all_paths_visualization', 10)
        self.timer = self.create_timer(0.01, self.timer_callback)

    def get_iteration_color(self, agent_id, iteration):
        """에이전트별 고유 색상 + 반복별 투명도"""
        # 에이전트별 고유 색상 (HSV 기반)
        hue = (agent_id * 0.3) % 1.0  # 색상 간격
        
        # HSV to RGB 간단 변환
        if hue < 1/3:
            r, g, b = 1.0, hue*3, 0.0
        elif hue < 2/3:
            r, g, b = 2.0-hue*3, 1.0, 0.0
        else:
            r, g, b = 0.0, 3.0-hue*3, hue*3-2.0
        
        # 반복별 투명도 (초기 반복은 흐릿하게, 최종 반복은 진하게)
        alpha = 0.2 + 0.6 * (iteration / self.max_iter)
        
        return ColorRGBA(r=r, g=g, b=b, a=alpha)

    def timer_callback(self):
        marker_array = MarkerArray()
        
        # 현재 시간까지의 경로만 표시
        current_time = self.time_steps[self.current_time_idx]
        
        # 모든 반복 주기를 동시에 처리
        for iteration in range(1, self.max_iter + 1):
            for agent_id in self.agents:
                # 해당 반복의 현재 시간까지 경로 데이터
                agent_df = self.df[
                    (self.df['agent'] == agent_id) &
                    (self.df['replan_iter'] == iteration) &
                    (self.df['time'] <= current_time)
                ].sort_values('time')
                
                if len(agent_df) >= 2:
                    # 경로 라인 마커
                    line_marker = Marker()
                    line_marker.header.frame_id = "map"
                    line_marker.header.stamp = self.get_clock().now().to_msg()
                    line_marker.ns = f"path_agent_{agent_id}_iter_{iteration}"
                    line_marker.id = int(agent_id * 100 + iteration)  # 고유 ID
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
        
        # 최신 반복의 현재 위치 마커들
        for agent_id in self.agents:
            current_pos = self.df[
                (self.df['agent'] == agent_id) &
                (self.df['replan_iter'] == self.max_iter) &
                (self.df['time'] == current_time)
            ]
            
            if not current_pos.empty:
                row = current_pos.iloc[0]
                sphere_marker = Marker()
                sphere_marker.header.frame_id = "map"
                sphere_marker.header.stamp = self.get_clock().now().to_msg()
                sphere_marker.ns = f"current_agent_{agent_id}"
                sphere_marker.id = int(agent_id * 1000 + 999)
                sphere_marker.type = Marker.SPHERE
                sphere_marker.action = Marker.ADD
                sphere_marker.pose.position.x = float(row['x'])
                sphere_marker.pose.position.y = float(row['y'])
                sphere_marker.pose.position.z = float(row['z'])
                sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = 0.3
                
                # 현재 위치는 불투명하게
                color = self.get_iteration_color(agent_id, self.max_iter)
                color.a = 1.0
                sphere_marker.color = color
                
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
    node = AllIterationsVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
