#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ProjectionTester:
    """Test 3D to 2D projection with your camera parameters"""
    
    def __init__(self):
        # YOUR camera parameters
        self.fx = 1397.22
        self.fy = 1397.22
        self.cx = 960.0
        self.cy = 540.0
        
        # Camera matrix K
        self.K = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1      ]
        ])
        
        self.image_width = 1920
        self.image_height = 1080
        
        print("Camera Parameters Loaded:")
        print(f"  fx = {self.fx:.2f} pixels")
        print(f"  fy = {self.fy:.2f} pixels")
        print(f"  cx = {self.cx:.2f} pixels")
        print(f"  cy = {self.cy:.2f} pixels")
        print(f"  Resolution = {self.image_width}x{self.image_height}")
        print()
    
    def project_point(self, point_3d):
        """
        Project a 3D point to 2D pixel coordinates
        
        Args:
            point_3d: [x, y, z] in camera frame (meters)
                     x: right, y: down, z: forward
        
        Returns:
            pixel: [u, v] in image (pixels)
        """
        # Project using camera matrix
        point_3d = np.array(point_3d)
        pixel_homogeneous = self.K @ point_3d
        
        # Normalize by depth (z coordinate)
        pixel = pixel_homogeneous[:2] / pixel_homogeneous[2]
        
        return pixel
    
    def is_in_image(self, pixel):
        """Check if pixel is visible in image"""
        u, v = pixel
        return (0 <= u < self.image_width and 
                0 <= v < self.image_height)
    
    def test_basic_projection(self):
        """Test basic projection with simple points"""
        print("=" * 70)
        print("TEST 1: Basic Projection")
        print("=" * 70)
        
        test_points = [
            ([0, 0, 5], "5m straight ahead, center"),
            ([1, 0, 5], "5m ahead, 1m right"),
            ([-1, 0, 5], "5m ahead, 1m left"),
            ([0, 1, 5], "5m ahead, 1m down"),
            ([0, -1, 5], "5m ahead, 1m up"),
            ([0, 0, 10], "10m straight ahead"),
        ]
        
        for point_3d, description in test_points:
            pixel = self.project_point(point_3d)
            visible = self.is_in_image(pixel)
            
            status = "✓ VISIBLE" if visible else "✗ OUT OF VIEW"
            
            print(f"Point: {point_3d}")
            print(f"  Description: {description}")
            print(f"  Pixel: ({pixel[0]:.1f}, {pixel[1]:.1f}) {status}")
            print()
    
    def test_grid_projection(self):
        """Test projection of a grid of points"""
        print("=" * 70)
        print("TEST 2: Grid Projection")
        print("=" * 70)
        
        visible_count = 0
        total_count = 0
        
        for x in np.linspace(-3, 3, 7):  # Left to right
            for y in np.linspace(-2, 2, 5):  # Up to down
                for z in [3, 5, 10]:  # Different depths
                    total_count += 1
                    point_3d = [x, y, z]
                    pixel = self.project_point(point_3d)
                    
                    if self.is_in_image(pixel):
                        visible_count += 1
        
        print(f"Grid: 7x5 points at 3 depths = {total_count} total points")
        print(f"Visible: {visible_count}/{total_count} points")
        print(f"Coverage: {visible_count/total_count*100:.1f}%")
        print()
    
    def visualize_frustum(self):
        """Visualize camera viewing frustum in 3D"""
        print("=" * 70)
        print("TEST 3: Visualizing Camera Frustum")
        print("=" * 70)
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Camera at origin
        ax.scatter(0, 0, 0, c='red', s=300, marker='^', 
                  label='Camera', edgecolors='black', linewidths=2)
        
        # Draw frustum at different depths
        colors = ['blue', 'green', 'orange']
        depths = [3, 5, 10]
        
        for depth, color in zip(depths, colors):
            # Four corners of image at this depth
            corners_pixel = [
                [0, 0],                          # Top-left
                [self.image_width, 0],           # Top-right
                [self.image_width, self.image_height],  # Bottom-right
                [0, self.image_height]           # Bottom-left
            ]
            
            # Back-project to 3D
            corners_3d = []
            for u, v in corners_pixel:
                # Create ray through pixel
                ray = np.linalg.inv(self.K) @ np.array([u, v, 1])
                # Scale ray to desired depth
                point_3d = ray * depth / ray[2]
                corners_3d.append(point_3d)
            
            corners_3d = np.array(corners_3d)
            
            # Draw rectangle at this depth
            for i in range(4):
                j = (i + 1) % 4
                ax.plot([corners_3d[i, 0], corners_3d[j, 0]],
                       [corners_3d[i, 1], corners_3d[j, 1]],
                       [corners_3d[i, 2], corners_3d[j, 2]], 
                       c=color, linewidth=2, label=f'{depth}m' if i == 0 else '')
            
            # Draw lines from camera to corners
            for corner in corners_3d:
                ax.plot([0, corner[0]], [0, corner[1]], [0, corner[2]], 
                       c=color, linestyle='--', alpha=0.3, linewidth=1)
        
        # Labels and formatting
        ax.set_xlabel('X (m) - Right', fontsize=12)
        ax.set_ylabel('Y (m) - Down', fontsize=12)
        ax.set_zlabel('Z (m) - Forward', fontsize=12)
        ax.set_title('Camera Viewing Frustum', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        
        # Set equal aspect ratio
        max_range = 10
        ax.set_xlim([-max_range/2, max_range/2])
        ax.set_ylim([-max_range/2, max_range/2])
        ax.set_zlim([0, max_range])
        
        plt.tight_layout()
        plt.savefig('camera_frustum_module1.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: camera_frustum_module1.png")
        plt.show()
    
    def calculate_fov(self):
        """Calculate field of view"""
        print("=" * 70)
        print("TEST 4: Field of View Calculation")
        print("=" * 70)
        
        # Horizontal FOV
        fov_horizontal = 2 * np.arctan(self.image_width / (2 * self.fx))
        fov_horizontal_deg = np.degrees(fov_horizontal)
        
        # Vertical FOV
        fov_vertical = 2 * np.arctan(self.image_height / (2 * self.fy))
        fov_vertical_deg = np.degrees(fov_vertical)
        
        print(f"Horizontal FOV: {fov_horizontal_deg:.2f}°")
        print(f"Vertical FOV: {fov_vertical_deg:.2f}°")
        print()
        
        # Visible area at different distances
        print("Visible area at different distances:")
        for distance in [1, 3, 5, 10, 20]:
            width = 2 * distance * np.tan(fov_horizontal / 2)
            height = 2 * distance * np.tan(fov_vertical / 2)
            print(f"  At {distance:2d}m: {width:.2f}m wide × {height:.2f}m tall")
        print()


def main():
    print("\n" + "=" * 70)
    print("MODULE 1: CAMERA PROJECTION TESTING")
    print("=" * 70 + "\n")
    
    tester = ProjectionTester()
    
    # Run all tests
    tester.calculate_fov()
    tester.test_basic_projection()
    tester.test_grid_projection()
    tester.visualize_frustum()
    
    print("=" * 70)
    print("✓ All tests complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
