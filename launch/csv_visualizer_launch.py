from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mapf_visualizer_3d',
            namespace='',
            executable='csv_visualizer',
            name='sim'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rivz2',
            arguments=['-d', '/home/kty/ros2_ws/src/mapf_visualizer_3d/rviz/mapf_visualizer.rviz'],
            output='screen'
        )
    ])