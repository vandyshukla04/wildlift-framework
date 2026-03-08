#!/usr/bin/env python3
"""
Simple Point Cloud Visualizer using Viser

Usage:
    python visualize_pointcloud.py <path_to_pointcloud>
    python visualize_pointcloud.py <path_to_rgbd.npz> --frame 10
    python visualize_pointcloud.py <mega_sam_output_dir> --frame 0

Supported formats:
    - .ply (PLY format)
    - .pcd (PCD format)
    - .npy (NumPy array with shape [N, 3] or [N, 6] for XYZ+RGB)
    - .npz (NumPy archive with 'points' and optionally 'colors' keys)
    - .npz (RGBD sequence with 'images', 'depths', 'intrinsic', 'cam_c2w')
    - .xyz (XYZ text format)
    - .pts (PTS format)
    - Directory with mega-sam processed_frames (frame_*.npz with background_points, instance_*_points)
"""

import argparse
import numpy as np
import viser
import time
import glob
from pathlib import Path


def unproject_depth(depth: np.ndarray, intrinsic: np.ndarray, c2w: np.ndarray = None) -> np.ndarray:
    """
    Unproject depth map to 3D points.

    Args:
        depth: [H, W] depth map
        intrinsic: [3, 3] camera intrinsic matrix
        c2w: [4, 4] camera-to-world transformation (optional)

    Returns:
        points: [H*W, 3] 3D points
    """
    H, W = depth.shape

    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()

    # Get camera parameters
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Unproject to camera space
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_cam = np.stack([x, y, z], axis=-1)

    # Transform to world space if c2w provided
    if c2w is not None:
        points_cam_h = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=-1)
        points_world = (c2w @ points_cam_h.T).T[:, :3]
        return points_world

    return points_cam


def check_rgbd_sequence(filepath: Path) -> int | None:
    """
    Check if file is an RGBD sequence and return number of frames.
    Returns None if not an RGBD sequence.
    """
    if filepath.suffix.lower() != '.npz':
        return None

    data = np.load(filepath)
    keys = list(data.keys())
    if all(k in keys for k in ['images', 'depths', 'intrinsic', 'cam_c2w']):
        num_frames = len(data['images'])
        data.close()
        return num_frames
    data.close()
    return None


def load_rgbd_frame(filepath: Path, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a single frame from an RGBD sequence.

    Args:
        filepath: Path to the npz file
        frame_idx: Frame index to load

    Returns:
        points: [N, 3] 3D points
        colors: [N, 3] RGB colors normalized to [0, 1]
    """
    data = np.load(filepath)

    images = data['images']      # [num_frames, H, W, 3]
    depths = data['depths']      # [num_frames, H, W]
    intrinsic = data['intrinsic']  # [3, 3]
    cam_c2w = data['cam_c2w']    # [num_frames, 4, 4]

    num_frames = len(images)
    if frame_idx < 0 or frame_idx >= num_frames:
        raise ValueError(f"Frame index {frame_idx} out of range [0, {num_frames-1}]")

    depth = depths[frame_idx]
    image = images[frame_idx]
    c2w = cam_c2w[frame_idx]

    # Unproject depth to 3D
    points = unproject_depth(depth, intrinsic, c2w)

    # Get colors
    colors = image.reshape(-1, 3) / 255.0

    # Filter invalid depth (zero or too far)
    valid_mask = (depth.flatten() > 0) & (depth.flatten() < 100)
    points = points[valid_mask]
    colors = colors[valid_mask]

    return points.astype(np.float32), colors.astype(np.float32)


def check_megasam_directory(filepath: Path) -> list[Path] | None:
    """
    Check if filepath is a directory (or parent of) mega-sam processed_frames npz files.
    Returns sorted list of frame npz paths, or None.
    """
    if filepath.is_dir():
        # Check for processed_frames subdirectory
        pf = filepath / "processed_frames"
        if pf.is_dir():
            filepath = pf
        files = sorted(filepath.glob("frame_*.npz"))
        if not files:
            files = sorted(filepath.glob("*.npz"))
        if files:
            # Verify first file has the expected keys
            d = np.load(files[0], allow_pickle=True)
            if 'background_points' in d:
                d.close()
                return files
            d.close()
    return None


def load_megasam_frame(filepath: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Load a single mega-sam processed frame npz.

    Returns:
        points: [N, 3] combined points
        colors: [N, 3] combined colors (0-1)
        instances: list of dicts with keys: class, points, colors, bbox_center, bbox_dims
    """
    d = np.load(filepath, allow_pickle=True)

    all_points = [d['background_points']]
    all_colors = [d['background_colors']]
    instances = []

    num_inst = int(d['num_instances'])
    for i in range(num_inst):
        pts = d[f'instance_{i}_points']
        cols = d[f'instance_{i}_colors']
        all_points.append(pts)
        all_colors.append(cols)

        inst = {'points': pts, 'colors': cols}
        for key in ['class', 'score', 'track_id', 'persistent_id', 'bbox_center', 'bbox_dims']:
            k = f'instance_{i}_{key}'
            if k in d:
                inst[key] = d[k]
        instances.append(inst)

    points = np.vstack(all_points).astype(np.float32)
    colors = np.vstack(all_colors).astype(np.float32)
    colors = np.clip(colors, 0, 1)
    d.close()
    return points, colors, instances


def load_pointcloud(filepath: str) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load a point cloud from various file formats.

    Returns:
        points: np.ndarray of shape [N, 3] (XYZ coordinates)
        colors: np.ndarray of shape [N, 3] (RGB colors normalized to [0, 1]) or None
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    points = None
    colors = None

    if suffix == '.ply':
        try:
            import plyfile
            plydata = plyfile.PlyData.read(filepath)
            vertex = plydata['vertex']
            points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

            # Try to load colors
            if 'red' in vertex.data.dtype.names:
                colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T
                if colors.max() > 1:
                    colors = colors / 255.0
        except ImportError:
            # Fallback to trimesh
            import trimesh
            mesh = trimesh.load(filepath)
            if hasattr(mesh, 'vertices'):
                points = np.array(mesh.vertices)
                if hasattr(mesh, 'colors') and mesh.colors is not None:
                    colors = np.array(mesh.colors)[:, :3] / 255.0
            else:
                points = np.array(mesh)

    elif suffix == '.pcd':
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(filepath))
            points = np.asarray(pcd.points)
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
        except ImportError:
            raise ImportError("Please install open3d to load PCD files: pip install open3d")

    elif suffix == '.npy':
        data = np.load(filepath)
        if data.shape[1] == 3:
            points = data
        elif data.shape[1] == 6:
            points = data[:, :3]
            colors = data[:, 3:6]
            if colors.max() > 1:
                colors = colors / 255.0
        elif data.shape[1] > 6:
            points = data[:, :3]
            colors = data[:, 3:6]
            if colors.max() > 1:
                colors = colors / 255.0
        else:
            raise ValueError(f"Unexpected array shape: {data.shape}")

    elif suffix == '.npz':
        data = np.load(filepath, allow_pickle=True)
        # Check for mega-sam single frame format
        if 'background_points' in data:
            pts, cols, _ = load_megasam_frame(filepath)
            data.close()
            return pts, cols
        if 'points' in data:
            points = data['points']
        elif 'xyz' in data:
            points = data['xyz']
        elif 'pts' in data:
            points = data['pts']
        else:
            # Try first array
            keys = list(data.keys())
            points = data[keys[0]]

        # Try to get colors
        for color_key in ['colors', 'rgb', 'color']:
            if color_key in data:
                colors = data[color_key]
                if colors.max() > 1:
                    colors = colors / 255.0
                break

    elif suffix in ['.xyz', '.pts', '.txt']:
        data = np.loadtxt(filepath)
        if data.shape[1] >= 3:
            points = data[:, :3]
        if data.shape[1] >= 6:
            colors = data[:, 3:6]
            if colors.max() > 1:
                colors = colors / 255.0

    else:
        # Try trimesh as fallback
        try:
            import trimesh
            mesh = trimesh.load(filepath)
            if hasattr(mesh, 'vertices'):
                points = np.array(mesh.vertices)
                if hasattr(mesh, 'colors') and mesh.colors is not None:
                    colors = np.array(mesh.colors)[:, :3] / 255.0
            else:
                points = np.array(mesh)
        except Exception as e:
            raise ValueError(f"Unsupported file format: {suffix}. Error: {e}")

    if points is None:
        raise ValueError(f"Could not load points from {filepath}")

    return points.astype(np.float32), colors.astype(np.float32) if colors is not None else None


def _visualize_megasam(
    filepath: Path,
    frame_files: list[Path],
    port: int,
    point_size: float,
    default_color: tuple,
    frame_idx: int | None,
):
    """Visualize mega-sam processed_frames directory with frame scrubbing."""
    num_frames = len(frame_files)
    start_idx = frame_idx if frame_idx is not None else 0
    start_idx = max(0, min(start_idx, num_frames - 1))

    print(f"Detected mega-sam sequence with {num_frames} frames")
    print(f"Loading frame {start_idx}...")

    points, colors, instances = load_megasam_frame(frame_files[start_idx])
    print(f"Loaded {len(points):,} points, {len(instances)} instances")

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("-y")

    # State
    current_points = points
    current_colors = colors
    current_instances = instances
    pc_handle = server.scene.add_point_cloud(
        name="/pointcloud",
        points=points,
        colors=colors,
        point_size=point_size,
    )
    bbox_handles = []

    def _add_instance_bboxes(insts):
        nonlocal bbox_handles
        for h in bbox_handles:
            h.remove()
        bbox_handles = []
        if not show_bboxes.value:
            return
        # Generate distinct colors per instance
        cmap = plt.cm.tab10
        for i, inst in enumerate(insts):
            if 'bbox_center' not in inst or 'bbox_dims' not in inst:
                continue
            center = np.array(inst['bbox_center'], dtype=np.float64)
            dims = np.array(inst['bbox_dims'], dtype=np.float64)
            color = cmap(i % 10)[:3]
            label = str(inst.get('class', f'inst_{i}'))
            # Use a wireframe box via 12 line segments
            hx, hy, hz = dims / 2
            corners = np.array([
                [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
            ]) + center
            edges = [
                (0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7),
            ]
            for ei, (a, b) in enumerate(edges):
                seg = np.stack([corners[a], corners[b]])
                col_arr = np.tile(color, (2, 1)).astype(np.float32)
                h = server.scene.add_point_cloud(
                    name=f"/bbox_{i}/edge_{ei}",
                    points=seg.astype(np.float32),
                    colors=col_arr,
                    point_size=point_size * 3,
                )
                bbox_handles.append(h)
            # Label
            lh = server.scene.add_label(
                name=f"/bbox_{i}/label",
                text=f"{label} ({inst.get('score', 0):.2f})" if 'score' in inst else label,
                wxyz=(1, 0, 0, 0),
                position=center + np.array([0, -dims[1]/2 - 0.3, 0]),
            )
            bbox_handles.append(lh)

    import matplotlib.pyplot as plt

    # GUI
    with server.gui.add_folder("Frame Controls"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=start_idx,
        )
        frame_label = server.gui.add_text("Frame File", initial_value=frame_files[start_idx].stem)

    with server.gui.add_folder("Display"):
        ps_slider = server.gui.add_slider(
            "Point Size", min=0.0001, max=0.1, step=0.0001, initial_value=point_size,
        )
        show_bboxes = server.gui.add_checkbox("Show Bounding Boxes", initial_value=True)
        color_mode = server.gui.add_dropdown(
            "Color Mode",
            options=["Original", "Instance", "Height (Z)"],
            initial_value="Original",
        )

    with server.gui.add_folder("Info"):
        info_points = server.gui.add_text("Points", initial_value=f"{len(points):,}")
        info_instances = server.gui.add_text("Instances", initial_value=f"{len(instances)}")

    _add_instance_bboxes(instances)

    def _reload_frame(idx):
        nonlocal pc_handle, current_points, current_colors, current_instances
        pts, cols, insts = load_megasam_frame(frame_files[idx])
        current_points = pts
        current_colors = cols
        current_instances = insts
        pc_handle.remove()
        display_cols = _get_colors(color_mode.value, pts, cols, insts)
        pc_handle = server.scene.add_point_cloud(
            name="/pointcloud", points=pts, colors=display_cols, point_size=ps_slider.value,
        )
        _add_instance_bboxes(insts)
        frame_label.value = frame_files[idx].stem
        info_points.value = f"{len(pts):,}"
        info_instances.value = f"{len(insts)}"

    def _get_colors(mode, pts, cols, insts):
        if mode == "Original":
            return cols
        elif mode == "Instance":
            cmap = plt.cm.tab10
            result = np.full_like(cols, 0.5)  # gray for background
            offset = len(pts) - sum(len(inst['points']) for inst in insts)
            for i, inst in enumerate(insts):
                n = len(inst['points'])
                result[offset:offset+n] = np.array(cmap(i % 10)[:3], dtype=np.float32)
                offset += n
            return result
        elif mode == "Height (Z)":
            z = pts[:, 2]
            z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
            return plt.cm.viridis(z_norm)[:, :3].astype(np.float32)
        return cols

    @frame_slider.on_update
    def _(_):
        _reload_frame(int(frame_slider.value))

    @ps_slider.on_update
    def _(_):
        pc_handle.point_size = ps_slider.value

    @show_bboxes.on_update
    def _(_):
        _add_instance_bboxes(current_instances)

    @color_mode.on_update
    def _(_):
        nonlocal pc_handle
        pc_handle.remove()
        display_cols = _get_colors(color_mode.value, current_points, current_colors, current_instances)
        pc_handle = server.scene.add_point_cloud(
            name="/pointcloud", points=current_points, colors=display_cols, point_size=ps_slider.value,
        )

    print(f"\n{'='*50}")
    print(f"Mega-SAM Point Cloud Viewer")
    print(f"{'='*50}")
    print(f"Open in browser: http://localhost:{port}")
    print(f"Frames: {num_frames} | Instances in first frame: {len(instances)}")
    print(f"{'='*50}")
    print("Press Ctrl+C to exit")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")


def visualize_pointcloud(
    filepath: str,
    port: int = 8080,
    point_size: float = 0.005,
    default_color: tuple = (0.5, 0.5, 0.8),
    frame_idx: int | None = None,
):
    """
    Visualize a point cloud using viser.

    Args:
        filepath: Path to the point cloud file
        port: Port for the viser server
        point_size: Initial point size
        default_color: Default color if no colors are provided (RGB, 0-1 range)
        frame_idx: For RGBD sequences, which frame to visualize
    """
    filepath = Path(filepath)

    # Check if this is a mega-sam processed_frames directory
    megasam_frames = check_megasam_directory(filepath)
    if megasam_frames is not None:
        _visualize_megasam(filepath, megasam_frames, port, point_size, default_color, frame_idx)
        return

    # Check if this is an RGBD sequence
    num_frames = check_rgbd_sequence(filepath)

    if num_frames is not None:
        # It's an RGBD sequence
        print(f"Detected RGBD sequence with {num_frames} frames")

        if frame_idx is None:
            # Ask user which frame to visualize
            print(f"\nAvailable frames: 0 to {num_frames - 1}")
            try:
                frame_idx = int(input(f"Enter frame number to visualize [0-{num_frames-1}]: "))
            except (ValueError, EOFError):
                frame_idx = 0
                print(f"Using default frame 0")

        if frame_idx < 0 or frame_idx >= num_frames:
            print(f"Invalid frame index {frame_idx}, using frame 0")
            frame_idx = 0

        print(f"Loading frame {frame_idx}...")
        points, colors = load_rgbd_frame(filepath, frame_idx)
        print(f"Loaded {len(points):,} points from frame {frame_idx}")
    else:
        # Regular point cloud file
        print(f"Loading point cloud from: {filepath}")
        points, colors = load_pointcloud(str(filepath))
        print(f"Loaded {len(points):,} points")

    # Use default color if none provided
    if colors is None:
        print(f"No colors found, using default color: {default_color}")
        colors = np.tile(default_color, (len(points), 1)).astype(np.float32)

    # Filter out invalid points (NaN, Inf)
    valid_mask = np.isfinite(points).all(axis=1)
    if not valid_mask.all():
        num_invalid = (~valid_mask).sum()
        print(f"Filtering out {num_invalid:,} invalid points")
        points = points[valid_mask]
        colors = colors[valid_mask]

    # Create viser server
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("-y")

    # Add GUI controls
    with server.gui.add_folder("Point Cloud Controls"):
        point_size_slider = server.gui.add_slider(
            "Point Size",
            min=0.0001,
            max=0.1,
            step=0.0001,
            initial_value=point_size,
        )

        show_axes = server.gui.add_checkbox("Show Axes", initial_value=True)

        color_mode = server.gui.add_dropdown(
            "Color Mode",
            options=["Original", "Height (Z)", "Depth (X)", "Random"],
            initial_value="Original",
        )

        reset_view = server.gui.add_button("Reset View")

    # Add info text
    with server.gui.add_folder("Info"):
        server.gui.add_text("File", initial_value=filepath.name)
        if num_frames is not None:
            server.gui.add_text("Frame", initial_value=f"{frame_idx} / {num_frames - 1}")
        server.gui.add_text("Points", initial_value=f"{len(points):,}")

        # Compute bounds
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)
        extent = max_bounds - min_bounds
        server.gui.add_text("Extent X", initial_value=f"{float(extent[0]):.3f}")
        server.gui.add_text("Extent Y", initial_value=f"{float(extent[1]):.3f}")
        server.gui.add_text("Extent Z", initial_value=f"{float(extent[2]):.3f}")

    # Store original colors for switching
    original_colors = colors.copy()

    # Generate alternative color maps
    def get_height_colors():
        z = points[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        import matplotlib.pyplot as plt
        cmap = plt.cm.viridis
        return cmap(z_norm)[:, :3].astype(np.float32)

    def get_depth_colors():
        x = points[:, 0]
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        import matplotlib.pyplot as plt
        cmap = plt.cm.plasma
        return cmap(x_norm)[:, :3].astype(np.float32)

    def get_random_colors():
        return np.random.rand(len(points), 3).astype(np.float32)

    # Add point cloud to scene
    pc_handle = server.scene.add_point_cloud(
        name="/pointcloud",
        points=points,
        colors=colors,
        point_size=point_size,
    )

    # Add coordinate axes
    axes_handle = server.scene.add_frame(
        "/axes",
        axes_length=float(extent.max()) * 0.1,
        axes_radius=float(extent.max()) * 0.002,
    )

    # GUI callbacks
    @point_size_slider.on_update
    def _(_):
        pc_handle.point_size = point_size_slider.value

    @show_axes.on_update
    def _(_):
        axes_handle.visible = show_axes.value

    @color_mode.on_update
    def _(_):
        nonlocal pc_handle
        mode = color_mode.value

        if mode == "Original":
            new_colors = original_colors
        elif mode == "Height (Z)":
            new_colors = get_height_colors()
        elif mode == "Depth (X)":
            new_colors = get_depth_colors()
        elif mode == "Random":
            new_colors = get_random_colors()
        else:
            new_colors = original_colors

        # Remove old and add new point cloud (viser doesn't support color updates)
        pc_handle.remove()
        pc_handle = server.scene.add_point_cloud(
            name="/pointcloud",
            points=points,
            colors=new_colors,
            point_size=point_size_slider.value,
        )

    @reset_view.on_click
    def _(_):
        # Reset camera to default view
        for client in server.get_clients().values():
            center = (min_bounds + max_bounds) / 2
            distance = float(extent.max()) * 2
            client.camera.position = center + np.array([0, 0, distance])
            client.camera.look_at = center

    print(f"\n{'='*50}")
    print(f"Point Cloud Viewer")
    print(f"{'='*50}")
    print(f"Open in browser: http://localhost:{port}")
    if num_frames is not None:
        print(f"Frame: {frame_idx} / {num_frames - 1}")
    print(f"Points: {len(points):,}")
    print(f"Bounds: [{float(min_bounds[0]):.2f}, {float(min_bounds[1]):.2f}, {float(min_bounds[2]):.2f}] to [{float(max_bounds[0]):.2f}, {float(max_bounds[1]):.2f}, {float(max_bounds[2]):.2f}]")
    print(f"{'='*50}")
    print("Press Ctrl+C to exit")

    # Keep server running
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a point cloud using viser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to point cloud file (.ply, .pcd, .npy, .npz, .xyz, .pts) or mega-sam output directory"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the viser server (default: 8080)"
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.005,
        help="Initial point size (default: 0.005)"
    )
    parser.add_argument(
        "--color",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.8],
        metavar=("R", "G", "B"),
        help="Default RGB color if no colors in file (0-1 range, default: 0.5 0.5 0.8)"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="For RGBD sequences: frame index to visualize (will prompt if not specified)"
    )

    args = parser.parse_args()

    visualize_pointcloud(
        filepath=args.filepath,
        port=args.port,
        point_size=args.point_size,
        default_color=tuple(args.color),
        frame_idx=args.frame,
    )


if __name__ == "__main__":
    main()
