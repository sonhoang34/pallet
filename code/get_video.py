
# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import os

path_save_rgb = r'C:\Lap trinh\realsense\project\video5\rgb'
path_save_depth = r'C:\Lap trinh\realsense\project\video5\depth_raw'


def pre():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align


def get_frame(pipeline, align):
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    # Get aligned frames
    # aligned_depth_frame is a 640x480 depth image

    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Render images:
    #   depth align to color on left
    #   depth on right
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
        depth_image, alpha=0.03), cv2.COLORMAP_JET)

    return color_image, depth_image, depth_colormap


def get_next_id(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    existing = [int(f.split('.')[0]) for f in os.listdir(
        folder_path) if f.endswith('.png') and f.split('.')[0].isdigit()]
    return max(existing) + 1 if existing else 1


if __name__ == "__main__":
    os.makedirs(path_save_rgb, exist_ok=True)
    os.makedirs(path_save_depth, exist_ok=True)

    pipeline, align = pre()

    print("📸 Saving ALL frames at camera FPS (15 FPS)")
    print("Nhấn 'q' hoặc 'ESC' để thoát")

    img_id = get_next_id(path_save_rgb)

    try:
        while True:
            rgb, depth_raw, depth_color = get_frame(pipeline, align)
            if rgb is None:
                continue

            # ===== SAVE EVERY FRAME =====
            name_rgb = os.path.join(path_save_rgb, f"{img_id:06d}.png")
            name_depth = os.path.join(path_save_depth, f"{img_id:06d}.png")

            cv2.imwrite(name_rgb, rgb)
            cv2.imwrite(name_depth, depth_raw)

            img_id += 1

            # ===== SHOW (optional) =====
            cv2.imshow('RGB', rgb)
            cv2.imshow('Depth Color', depth_color)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
        print("Pipeline stopped.")
