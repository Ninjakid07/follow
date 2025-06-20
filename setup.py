import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'follow'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waypoint_publisher = follow.waypoint_publisher:main',
            'path_follower = follow.path_follower:main',
            'cylinder_detector = follow.horizontal_cylinder_detector:main',
            'mission_planner = follow.nav2_mission_planner:main',
        ],
    },
)
