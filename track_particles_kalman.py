import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def track_particles_kalman(
    video_path,
    min_area=100,
    max_area=500,
    blob_color=255,
    kalman_p=100.0,
    kalman_q=0.01,
    kalman_r=1.0,
    filter_method="gaussian",
    enable_color_filter=True
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

    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    if not ret:
        raise ValueError("Could not read the first frame of the video.")

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    centros = detectar_centroides(old_gray)
    if len(centros) == 0:
        raise ValueError("No particles detected in the first frame.")

    kalman_filters = [KalmanFilter2D(x, y) for x, y in centros]
    trajectories = [[kf.state[:2].copy()] for kf in kalman_filters]
    detected_positions = [[kf.state[:2].copy()] for kf in kalman_filters]
    skipped_counts = [0] * len(kalman_filters)
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        centros_actuales = detectar_centroides(frame_gray)
        p1 = np.array(centros_actuales, dtype=np.float32)
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

        nuevas_detecciones = [p1[i] for i in range(len(p1)) if i not in assigned_det]
        for det in nuevas_detecciones:
            new_kf = KalmanFilter2D(det[0], det[1])
            kalman_filters.append(new_kf)
            trajectories.append([det.copy()])
            detected_positions.append([det.copy()])
            skipped_counts.append(0)

        max_skips = 10
        for i in reversed(range(len(kalman_filters))):
            if skipped_counts[i] > max_skips:
                del kalman_filters[i]
                del trajectories[i]
                del detected_positions[i]
                del skipped_counts[i]

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

    cap.release()
    cv2.destroyAllWindows()

    return trajectories, detected_positions