from flask import Flask, jsonify, render_template, Response
import cv2
import threading
import os
import time
from datetime import datetime, timedelta
from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CORS 설정 추가

# YOLO 모델 로드 (best.pt 파일 사용)
model = YOLO('best.pt')

# 웹캠 인덱스
camera = cv2.VideoCapture(2)

# 이미지 저장 경로
img_path = './static/captured_image.jpg'

# 파일 접근을 보호하기 위한 Lock 생성
lock = threading.Lock()

# 클래스 이름을 정의 (YOLO 모델에서 사용하는 클래스 이름에 맞게 수정)
class_names = {
    0: 'level_1',
    1: 'level_2',
    2: 'level_3',
}

# 감지 상태를 저장하는 변수
detected = False
detection_time = time.time()  # 타이머 시작 시간
detection_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')  # 감지된 시간을 저장
tracking_time_limit = 5  # 객체가 유지되어야 하는 시간 (초)
is_tracking = False  # 객체 추적 활성화 여부
notification_threshold = 5  # 프론트엔드에 전달할 최소 경과 시간 (초)
last_notification_time = datetime.now() - timedelta(hours=5)  # 마지막 알림 시간을 현재 시간에서 5시간 이전으로 초기화
notification_interval = timedelta(hours=5)  # 알림을 보낼 최소 간격

# 클래스별 바운딩 박스 개수를 저장할 변수
class_counts = {0: 0, 1: 0, 2: 0}
# 주기적으로 웹캠에서 이미지를 캡처하고 YOLO로 처리하는 함수
def capture_image_periodically():
    global detected, detection_time, detection_timestamp, is_tracking, last_notification_time, class_counts
    while True:
        camera.grab()
        ret, frame = camera.retrieve()
        if ret:
            # YOLO 모델로 객체 감지
            results = model(frame)

            # 클래스별 바운딩 박스 개수 초기화
            class_counts = {0: 0, 1: 0, 2: 0}

            new_detected = False

            # 감지된 객체에 대한 클래스별 바운딩 박스 개수를 집계 및 바운딩 박스 그리기
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    label_id = int(box.cls)  # 클래스 레이블 (int로 변환)
                    if label_id in class_counts:
                        class_counts[label_id] += 1
                        new_detected = True

                    # 바운딩 박스 정보
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 좌상단 (x1, y1), 우하단 (x2, y2)
                    confidence = box.conf[0]  # 신뢰도 점수
                    label = class_names.get(label_id, 'Unknown')

                    # 바운딩 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 테두리
                    cv2.putText(frame, f'{label}', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 라벨과 신뢰도 점수

            # 객체가 감지되면 타이머 리셋 및 추적 시작
            if new_detected:
                if not detected:
                    detection_time = time.time()  # 타이머 리셋
                    detection_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')  # 감지된 시간 저장
                    is_tracking = True  # 객체 추적 시작
                detected = True

            # 객체 추적 중 일정 시간 경과 시 상태 업데이트
            if detected and is_tracking:
                elapsed_time = time.time() - detection_time
                if elapsed_time >= tracking_time_limit:
                    is_tracking = False  # 타이머 종료

            # 객체가 사라졌다면 감지 상태를 false로 업데이트
            if not new_detected:
                detected = False
                is_tracking = False

            # 감지된 이미지 저장
            with lock:
                cv2.imwrite(img_path, frame)

        time.sleep(0.1)  # 0.1초마다 이미지 캡처

# 별도의 스레드로 주기적으로 이미지 캡처 실행
thread = threading.Thread(target=capture_image_periodically)
thread.daemon = True
thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image', methods=['GET'])
def get_image():
    with lock:
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                image_data = f.read()
            return Response(image_data, mimetype='image/jpeg')
        else:
            return jsonify({'status': 'error', 'message': 'No image available.'})

# 프론트 : 객체 감지 상태(T/F로 같은 객체는 5시간 동안 알림 안가도록 설정), 라벨링 박스 개수 반환 api
@app.route('/status', methods=['GET'])
def get_detection_status():
    global detected, is_tracking, detection_time, last_notification_time, class_counts
    elapsed_time = round(time.time() - detection_time, 2) if detected else 0
    
    # 현재 시간이 마지막 알림 시간으로부터 5시간 이상 경과했는지 확인
    current_time = datetime.now()
    time_since_last_notification = (current_time - last_notification_time).total_seconds()
    should_notify = time_since_last_notification >= notification_interval.total_seconds()

    # 알림을 보낼 수 있는 상태인지 체크하고, 알림 전송 후 마지막 알림 시간 업데이트
    if detected and should_notify:
        last_notification_time = current_time  # 알림을 보낸 시간을 업데이트

    # 특정 시간 이상 경과한 경우에만 경과 시간을 프론트엔드로 보냄
    time_info = elapsed_time if elapsed_time >= notification_threshold and should_notify else 0

    status_info = {
        'detected': detected,
        'tracking': is_tracking,
        'time_since_detection': time_info,
        'should_notify': should_notify,  # 프론트엔드에서 알림 여부 확인 가능
        'class_counts': {
            'level_1': class_counts[0],
            'level_2': class_counts[1],
            'level_3': class_counts[2]
        }
    }
    return jsonify(status_info)

# 백엔드 : 객체 감지 상태(T/F로 같은 객체는 5시간 동안 알림 안가도록 설정), 현재 시간 반환 api
@app.route('/status_with_time', methods=['GET'])
def get_detection_status_with_time():
    global detected, is_tracking, detection_time, detection_timestamp, last_notification_time
    elapsed_time = round(time.time() - detection_time, 2) if detected else 0
    
    # 현재 시간이 마지막 알림 시간으로부터 5시간 이상 경과했는지 확인
    current_time = datetime.now()
    time_since_last_notification = (current_time - last_notification_time).total_seconds()
    should_notify = time_since_last_notification >= notification_interval.total_seconds()

    # 알림을 보낼 수 있는 상태인지 체크하고, 알림 전송 후 마지막 알림 시간 업데이트
    if detected and should_notify:
        last_notification_time = current_time  # 알림을 보낸 시간을 업데이트

    # 특정 시간 이상 경과한 경우에만 경과 시간을 반환
    time_info = elapsed_time if elapsed_time >= notification_threshold and should_notify else 0

    status_info = {
        'detected': detected,
        'tracking': is_tracking,
        'time_since_detection': time_info,
        'detection_timestamp': detection_timestamp if detected else None,
        'should_notify': should_notify  # 프론트엔드에서 알림 여부 확인 가능
    }
    return jsonify(status_info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
