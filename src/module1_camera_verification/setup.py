from setuptools import setup

package_name = 'module1_camera_verification'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ajay',
    maintainer_email='your_email@example.com',
    description='Module 1: Camera verification',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'camera_verifier = module1_camera_verification.camera_verifier:main',
        'projection_tester = module1_camera_verification.projection_tester:main',
        'feature_detector_comparison = module1_camera_verification.feature_detector_comparison:main',
        'feature_tracker = module1_camera_verification.feature_tracker:main',
        'imu_logger = module1_camera_verification.imu_logger:main',
        'imu_drift_demo = module1_camera_verification.imu_drift_demo:main',
        'cam_imu_sync = module1_camera_verification.cam_imu_sync:main',
    	],
	},
	)
