import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
import matplotlib.pyplot as plt

def track_particles_kalman(
    cap,
    min_area=100,
    max_area=500,
    blob_color=255,
    kalman_p=100.0,
    kalman_q=0.01,
    kalman_r=1.0,
    filter_method="gaussian",
    enable_color_filter=True,
    use_world_coords=False, 
    magnification=1.0, 
    pitch_x=1.0, 
    pitch_y=1.0
):
    class KalmanFilter2D:
        def __init__(self, x, y):
            self.state = np.array([x, y, 0, 0], dtype=np.float32)
            self.F = np.array([[1, 0, 1, 0],
                               [0, 1, 0, 1],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float32)
            self.H = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]], dtype=np.float32)
            self.P = np.eye(4, dtype=np.float32) * kalman_p
            self.Q = np.eye(4, dtype=np.float32) * kalman_q
            self.R = np.eye(2, dtype=np.float32) * kalman_r

        def predict(self):
            self.state = self.F @ self.state
            self.P = self.F @ self.P @ self.F.T + self.Q
            return self.state[:2]

        def update(self, measurement):
            z = np.array(measurement, dtype=np.float32)
            y = z - self.H @ self.state
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.state = self.state + K @ y
            I = np.eye(4)
            self.P = (I - K @ self.H) @ self.P

    def detectar_centroides(frame_gray):
        if filter_method == "gaussian":
            blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        elif filter_method == "bilateral":
            blurred = cv2.bilateralFilter(frame_gray, 9, 75, 75)
        else:
            blurred = frame_gray.copy()

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area
        params.filterByCircularity = False
        params.filterByInertia = True
        params.filterByConvexity = False
        params.filterByColor = enable_color_filter
        params.blobColor = blob_color

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(blurred)
        centroids = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        return centroids

    ret, old_frame = cap.read()
    if not ret:
        raise ValueError("Could not read the first frame of the video.")

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    centers = detectar_centroides(old_gray)
    if len(centers) == 0:
        raise ValueError("No particles detected in the first frame.")

    kalman_filters = [KalmanFilter2D(x, y) for x, y in centers]
    trajectories = [[kf.state[:2].copy()] for kf in kalman_filters]
    detected_positions = [[kf.state[:2].copy()] for kf in kalman_filters]
    skipped_counts = [0] * len(kalman_filters)
    mask = np.zeros_like(old_frame)
    archived_trajectories = []
    archived_detected_positions = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_centers = detectar_centroides(frame_gray)
        p1 = np.array(current_centers, dtype=np.float32)
        predictions = np.array([kf.predict() for kf in kalman_filters])

        if len(p1) == 0:
            for i, kf in enumerate(kalman_filters):
                trajectories[i].append(kf.state[:2].copy())
                detected_positions[i].append(detected_positions[i][-1])
                skipped_counts[i] += 1
            continue

        dist_matrix = np.linalg.norm(predictions[:, None, :] - p1[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        max_dist = 45
        assigned_pred, assigned_det = set(), set()
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] < max_dist:
                kalman_filters[r].update(p1[c])
                assigned_pred.add(r)
                assigned_det.add(c)
                detected_positions[r].append(p1[c].copy())
                skipped_counts[r] = 0
            else:
                detected_positions[r].append(detected_positions[r][-1])
                skipped_counts[r] += 1

        for i, kf in enumerate(kalman_filters):
            if i not in assigned_pred:
                detected_positions[i].append(detected_positions[i][-1])
                skipped_counts[i] += 1

        new_detections = [p1[i] for i in range(len(p1)) if i not in assigned_det]
        for det in new_detections:
            new_kf = KalmanFilter2D(det[0], det[1])
            kalman_filters.append(new_kf)
            trajectories.append([det.copy()])
            detected_positions.append([det.copy()])
            skipped_counts.append(0)

        max_skips = 10
        for i in reversed(range(len(kalman_filters))):
            if skipped_counts[i] > max_skips:
                archived_trajectories.append(trajectories[i])
                archived_detected_positions.append(detected_positions[i])
                del kalman_filters[i]
                del trajectories[i]
                del detected_positions[i]
                del skipped_counts[i]
        
        frame_number += 1
        all_trajectories = trajectories + archived_trajectories
        all_detected_positions = detected_positions + archived_detected_positions

        for i, kf in enumerate(kalman_filters):
            trajectories[i].append(kf.state[:2].copy())

        for traj in trajectories:
            if len(traj) > 1:
                pts = np.array(traj[-2:], dtype=int)
                cv2.line(mask, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2)
                cv2.circle(frame, tuple(pts[1]), 3, (0, 0, 255), -1)

        output = cv2.add(frame, mask)
        scale = 0.7
        resized_output = cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Kalman Tracking", resized_output)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    # --- Trajentories plot ---
    fig_main, ax_main = plt.subplots(figsize=(8, 6))
    
    for i, traj in enumerate(all_trajectories):
        traj_array = np.array(traj)
        if use_world_coords:
            pixel_size_x_um = pitch_x / magnification
            pixel_size_y_um = pitch_y / magnification
            traj_array[:, 0] *= pixel_size_x_um
            traj_array[:, 1] *= pixel_size_y_um

        if len(traj_array) > 1:
            ax_main.plot(traj_array[:, 0], traj_array[:, 1], label=f'Particle {i}')
            ax_main.scatter(traj_array[0, 0], traj_array[0, 1], marker='o', c='green')  # start
            ax_main.scatter(traj_array[-1, 0], traj_array[-1, 1], marker='x', c='red')  # final
            ax_main.text(traj_array[-1, 0] + 10, traj_array[-1, 1] - 10, f'{i}', 
                        color='blue', fontsize=10, weight='bold')

    ax_main.invert_yaxis()
    ax_main.set_title("Trajectories of Tracked Particles")
    if use_world_coords:
        ax_main.set_xlabel("X (µm)")
        ax_main.set_ylabel("Y (µm)")
    else:
        ax_main.set_xlabel("X (pixels)")
        ax_main.set_ylabel("Y (pixels)")
    ax_main.grid(True)
    fig_main.tight_layout()
    plt.show(block=False)

    # --- coordinates of detected positions ---
    all_positions = []

    for particle_id, traj in enumerate(all_detected_positions ):
        for frame_idx, pos in enumerate(traj):
            all_positions.append({
                "particle_id": particle_id,
                "frame": frame_idx,
                "x": float(pos[0]),
                "y": float(pos[1])
            })

    df_full = pd.DataFrame(all_positions)
    df_full["x"] = df_full["x"].round(2)
    df_full["y"] = df_full["y"].round(2)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time_s = 1 / fps if fps > 0 else None
    if frame_time_s is not None:
        df_full["time_s"] = df_full["frame"] * frame_time_s
    df_pivot = df_full.pivot(index="frame", columns="particle_id", values=["x", "y"])
    df_pivot = df_pivot.swaplevel(axis=1).sort_index(axis=1, level=0)
    df_pivot.columns = [f"P{pid}_{coord}" for pid, coord in df_pivot.columns]
    df_positions_vector = df_pivot.reset_index().rename(columns={"frame": "frames"})
    if frame_time_s is not None:
        df_positions_vector["time_s"] = df_positions_vector["frames"] * frame_time_s
        cols = ["frames", "time_s"] + [c for c in df_positions_vector.columns if c not in ["frames", "time_s"]]
        df_positions_vector = df_positions_vector[cols]

    if use_world_coords:
        
        pixel_size_x_um = pitch_x / magnification
        pixel_size_y_um = pitch_y / magnification

        for col in df_positions_vector.columns:
            if col.endswith("_x"):
                df_positions_vector[col] *= pixel_size_x_um
            elif col.endswith("_y"):
                df_positions_vector[col] *= pixel_size_y_um

    return trajectories, detected_positions, df_full, df_positions_vector