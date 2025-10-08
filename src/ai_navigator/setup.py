from setuptools import find_packages, setup

package_name = 'ai_navigator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mcfb',
    maintainer_email='mcfb@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        #'ai_camera_sub = ai_navigator.ai_camera_sub:main',
        #'navigate_to_point = ai_navigator.navigate_to_point:main',
        #'point_to_point_nav = ai_navigator.point_to_point_nav:main',
        #'autonomous_navigation = ai_navigator.autonomous_navigation:main',
        'autonomous_navigation = ai_navigator.main:main'
        ],
    },
)
