import pyrealsense2 as rs
import numpy as np
import cv2
import scipy.io
import datetime

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream color and depth
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# Enable streams
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth Scale: {depth_scale}")

# Create an align object to align depth to color frame
align_to = rs.stream.color
align = rs.align(align_to)

# Create a colorizer object for .ply generation
colorizer = rs.colorizer()

try:
    while True:
        # Wait for frames and align
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap to depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))

        # Display images
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Align Example', 1280 * 2, 720)
        cv2.imshow('Align Example', images)

        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC to quit
            break
        if key == ord('s'):  # 'p' to save RGB-D data as .mat
            Depth = np.array(depth_image, dtype='f4')
            ColorBRG = np.array(color_image / 255, dtype='f4')
            Color = ColorBRG.copy()
            Color[:, :, 0] = ColorBRG[:, :, 2]
            Color[:, :, 2] = ColorBRG[:, :, 0]
            Dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            scipy.io.savemat(f'RGBD_D455_{Dt}.mat', mdict={'Depth': Depth, 'Color': Color})
            print(f"Saved RGBD data as RGBD_D455_{Dt}.mat")
        if key == ord('s'):  # 's' to save point cloud as .ply
            Dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            colorized = colorizer.process(aligned_frames)
            ply = rs.save_to_ply(f"RGBD_D455_{Dt}.ply")
            ply.set_option(rs.save_to_ply.option_ply_binary, True)
            ply.set_option(rs.save_to_ply.option_ply_normals, True)
            ply.process(colorized)
            print(f"Saved point cloud as RGBD_D455_{Dt}.ply")

finally:
    # Stop the pipeline and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Pipeline stopped.")
