from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'module3_vio'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ajay',
    maintainer_email='ajay@drone.dev',
    description='Phase 3: Visual-Inertial Odometry',
    license='MIT',
    entry_points={
        'console_scripts': [
            # Step 7 — OpenVINS study
            'sensor_bridge    = module3_vio.step7_sensor_bridge:main',
            'vio_comparator   = module3_vio.step7_vio_comparator:main',
            # Step 8 — Loosely-coupled EKF VIO
            'ekf_vio          = module3_vio.step8_ekf_vio:main',
            # Step 9 — Tightly-coupled MSCKF
            'msckf_vio        = module3_vio.step9_msckf:main',
            # Step 10 — Multi-camera MSCKF (metric scale via front+back stereo)
            'multicam_msckf   = module3_vio.step10_multicam_msckf:main',
        ],
    },
)
