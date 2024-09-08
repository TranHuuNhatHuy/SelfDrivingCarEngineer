# THE LIDAR SENSOR

## I. The roles of Lidar in AD

## II. Comparison - Lidar, Radar and Camera

1. Range:
- LiDAR: Detects objects from a few meters to over 200m, but struggles with close distances.
- Radar: Detects objects from less than 1m to over 200m, depending on range type (long, mid, short).
- Cameras: Mono cameras can't reliably measure distance; stereo cameras can, but only up to 80m.

2. Spatial Resolution:
- LiDAR: High spatial resolution (~0.1°) allows detailed 3D scans.
- Radar: Poor at resolving small features, especially at longer distances.
- Cameras: Resolution depends on optics, pixel size, and signal-to-noise ratio. Loses detail in poor lighting or when objects are too far away.

3. Robustness in Darkness:
- LiDAR & Radar: Excellent performance at night since they are active sensors.
- Cameras: Reduced performance at night due to reliance on ambient light.

4. Robustness in Adverse Weather:
- Radar: Not significantly affected by rain, snow, fog, or sand.
- LiDAR & Cameras: Performance degrades in adverse weather due to their reliance on optical systems.

5. Object Classification:
- Cameras: Best for classifying objects like vehicles, pedestrians, and signs.
- LiDAR: Can classify objects to a certain extent using 3D point clouds.
- Radar: Limited object classification ability.

6. Perceiving 2D Structures:
- Cameras: The only sensor capable of interpreting 2D structures like lane markings, speed signs, and traffic lights.

7. Measuring Speed:
- Radar: Directly measures object velocity using Doppler shift.
- LiDAR: Estimates speed using successive distance measurements, less accurate.
- Cameras: Can measure time to collision through object displacement on the image plane.

8. System Cost:
- Radar & Mono Cameras: Affordable, with radar systems being compact.
- Stereo Cameras: More expensive due to higher hardware costs.
- LiDAR: Prices have dropped dramatically, with potential to fall below US$500 soon.

9. Package Size:
- Radar & Mono Cameras: Easy to integrate into vehicles.
- Stereo Cameras: Bulky and harder to integrate, sometimes obstructing the driver's view.
- LiDAR: Available in various sizes; 360° scanning models are large but new solid-state versions are smaller.

10. Computational Requirements:
- LiDAR & Radar: Low computational demand.
- Cameras: Require significant processing to extract useful data, increasing system costs.

| | Range measurement | Spatial Resolution | Robustness in daylight | Robustness in darkness | Robustness in rain/snow/fog | Classification in objects | Perceiving 2D structures | Measure speed / TTC | Package size | System cost | Computational requirements |
| - | - | - | - | - | - | - | - | - | - | - | - |
| **Camera** | - | + | + | - | - | ++ | ++ | + | + | + | - |
| **Radar** | ++ | - | ++ | ++ | ++ | - | - | ++ | + | + | + |
| **Lidar** | + | ++ | + | ++ | + | + | - | + | - | - | ++ |

## III. Waymo Lidar system & dataset

### 1. Lidar system

![alt text](image.png)

The LiDAR sensors can be categorized into two broad groups:

1. **Perimeter Lidar**
- Vertical field of vision: -90&deg; to +30&deg;
- Range: 0-20 m.
- Those parameters are only reflected in Waymo Dataset. The actual parameters in real-life might be higher.

![alt text](image-1.png)

Waymo Front-Left LiDAR 3D Point-Cloud

2. **360 Lidar**
- Vertical field of vision: -17.6&deg; to +2.4&deg;
- Range: 75 m.
- Those parameters are only reflected in Waymo Dataset. The actual parameters in real-life might be higher.

![alt text](image-2.png)

Waymo Top LiDAR 3D Point-Cloud

Two aspects are noteworthy here: (1) the distance between adjacent scanner lines increases with growing distance and (b) the area in the direct circumference of the vehicle does not contain any 3d points. Both observations can be easily explained by a look at the geometry of the sensor-vehicle setup:

![alt text](image-3.png)

Directly in front of the vehicle, there is a large gap in perception ("blind spot") due to the occlusion of the laser beam by the vehicle. Also, it can be seen that the gap between adjacent beams is widening with distance due to the fixed angle in which laser diodes are positioned vertically.

