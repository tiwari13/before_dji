
"""
Launch file for Enhanced Smart Navigator
Supports multiple configurations and modes
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    """Generate launch description for Smart Navigator"""
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('ai_navigator'),
            'config',
            'default_config.yaml'
        ]),
        description='Path to configuration file'
    )
    
    flight_mode_arg = DeclareLaunchArgument(
        'flight_mode',
        default_value='POSITION',
        description='Flight mode: POSITION, SPORT, CINEMATIC, TRIPOD, ACTIVETRACK'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='INFO',
        description='Logging level: DEBUG, INFO, WARNING, ERROR'
    )
    
    use_simulation_arg = DeclareLaunchArgument(
        'use_simulation',
        default_value='true',
        description='Whether to use simulation mode'
    )
    
    # Smart Navigator node
    smart_navigator_node = Node(
        package='ai_navigator',
        executable='smart_navigator',
        name='smart_navigator',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'flight_mode': LaunchConfiguration('flight_mode'),
                'use_simulation': LaunchConfiguration('use_simulation'),
            }
        ],
        arguments=[
            '--log-level', LaunchConfiguration('log_level'),
            '--config', LaunchConfiguration('config_file'),
            '--mode', LaunchConfiguration('flight_mode')
        ],
        output='screen',
        emulate_tty=True,
        respawn=False,
        respawn_delay=2.0
    )
    
    return LaunchDescription([
        config_file_arg,
        flight_mode_arg,
        log_level_arg,
        use_simulation_arg,
        
        LogInfo(msg=['Starting Enhanced Smart Navigator with config: ', 
                    LaunchConfiguration('config_file')]),
        LogInfo(msg=['Flight mode: ', LaunchConfiguration('flight_mode')]),
        
        smart_navigator_node,
    ])
