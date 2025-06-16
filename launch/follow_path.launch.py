from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    waypoint_publisher_node = Node(
        package='follow',
        executable='waypoint_publisher',
        name='waypoint_publisher_node',
        output='screen'
    )
    
    path_follower_node = Node(
        package='follow',
        executable='path_follower',
        name='path_follower_node',
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(waypoint_publisher_node)
    ld.add_action(path_follower_node)
    
    return ld