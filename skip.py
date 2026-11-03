import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def calc_angle(a,b,c):
    vec1 = np.array([a.x - b.x, a.y - b.y])
    vec2 = np.array([c.x - b.x, c.y - b.y])
    angle = np.arccos(np.clip(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)), -1.0, 1.0))
    return np.degrees(angle)

count = 0
state = 0   # 0：等待下降；1：等待回升
min_interval = 10
frame_id = 0
last_count_frame = 0

# ========== 可调参数（重点） ==========
drop_thresh = 4    # 角度变化幅度阈值，越小越灵敏
# 觉得还是不计数 → 改成3
# 晃一下就乱加数字 → 改成5、6
# =====================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 摄像头打不开")
    exit()
win_name = "跳绳计数（小幅连续跳专用）"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

prev_angle = 155

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
        ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
        knee_angle = calc_angle(hip, knee, ankle)

        delta = knee_angle - prev_angle
        if state == 0:
            # 等待角度向下变小（屈膝），下降幅度超过阈值
            if delta < -drop_thresh:
                state = 1
        elif state == 1:
            # 等待角度向上回弹（伸直）
            if delta > drop_thresh:
                if frame_id - last_count_frame > min_interval:
                    count += 1
                    last_count_frame = frame_id
                state = 0

        prev_angle = knee_angle

        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f"Count: {count}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,255,0),3)
        cv2.putText(frame, f"KneeAngle:{knee_angle:.1f}", (30,110),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame, f"Delta:{delta:.1f}", (30,150),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

    cv2.imshow(win_name, frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ 总计数：", count)