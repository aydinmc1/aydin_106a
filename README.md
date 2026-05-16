# EECS106A Operation Surgeon

ROS 2 Python package for detecting green game-piece nubs with a RealSense camera, saving a stable snapshot of their positions, and sending a UR robot through a MoveIt path to pick/release them.

## Main Pieces

- `green_nub_detector_node_v2`: finds green nubs in RGB images, uses aligned depth to estimate 3D positions, publishes live poses, debug images, and ArUco marker TF.
- `nub_snapshot_node_v2`: records live nub detections for a few seconds, clusters them, then publishes a clean snapshot.
- `nub_path_executor_node_v2`: plans a MoveIt Cartesian path through snapshot nubs and calls the Pi gripper over HTTP.
- `aruco_home_node_v2`: moves the robot to a home pose relative to the detected ArUco marker.

## Build

From the workspace root:

```bash
colcon build --packages-select aydin_v2
source install/setup.bash
```

## Run

Launch the full pipeline:

```bash
ros2 launch aydin_v2 bringup.launch.py
```

Useful launch options:

```bash
ros2 launch aydin_v2 bringup.launch.py launch_camera:=false
ros2 launch aydin_v2 bringup.launch.py launch_path_executor:=false
ros2 launch aydin_v2 bringup.launch.py enable_tuning_window:=false
```

## Common Commands

Take a nub snapshot:

```bash
ros2 service call /take_nub_snapshot_v2 std_srvs/srv/Trigger
```

Plan and execute the nub path:

```bash
ros2 service call /plan_execute_nub_path_v2 std_srvs/srv/Trigger
```

Move home relative to the ArUco marker:

```bash
ros2 service call /home_to_aruco_v2 std_srvs/srv/Trigger
```

## Important Topics

- `/aydin_v2/green_nubs/debug_image_v2`: annotated camera image
- `/aydin_v2/live_nubs_v2/poses`: live detected nub poses
- `/aydin_v2/nub_snapshot_v2/poses`: clustered snapshot poses
- `/aydin_v2/nub_path_v2/path`: planned path visualization

## Config

Main ROS parameters live in `aydin_v2/config/params.yaml`.

Green threshold tuning is saved in `green_detection_params_v2.json`. If the OpenCV tuning window is enabled, press `s` or use the save control to update it.

## Notes

This package expects a RealSense aligned depth stream, camera TF into `base_link`, MoveIt services/actions, and the Pi gripper endpoint configured by `gripper_pi_url`.

`EXAMPLE_OF_GRIPPER_USAGE.py` is an older reference script for basic gripper/job-queue usage.
