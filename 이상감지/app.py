from flask import Flask, jsonify, render_template, Response
import cv2
import threading
import os
import time
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
last_detection_time = time.strftime('%Y-%m-%d %H:%M:%S')  # 상태 변경 시간을 저장

# 주기적으로 웹캠에서 이미지를 캡처하고 YOLO로 처리하는 함수
def capture_image_periodically():
    global detected, last_detection_time
    while True:
        # 버퍼를 제거하기 위해 grab()을 먼저 호출
        camera.grab()

        # 최신 프레임을 가져오기 위해 retrieve() 호출
        ret, frame = camera.retrieve()

        if ret:
            # YOLO 모델로 객체 감지
            results = model(frame)
            new_detected = False  # 초기화

            # 객체 감지된 결과를 이미지에 표시
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    label_id = int(box.cls)

                    # 클래스에 따른 색깔 지정
                    if label_id == 0:  # Level_1
                        color = (0, 255, 255)  # 노란색 (BGR)

                    elif label_id == 1:  # Level_2
                        color = (0, 0, 255)  # 빨간색 (BGR)

                    elif label_id == 2:  # Level_3
                        color = (255, 0, 0)  # 파란색 (BGR)

                    # 클래스 이름을 가져옴 (클래스 레이블에 해당하는 이름)
                    label = class_names.get(label_id, 'Unknown')

                    # 바운딩 박스와 클래스 이름 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    new_detected = True  # 객체가 감지되면 상태를 True로 설정
                    print(f'Detected: {label} at [{x1}, {y1}, {x2}, {y2}]')  # 감지 로그 출력

            # 감지 상태가 변경되었을 때만 업데이트
            if new_detected != detected:
                detected = new_detected
                last_detection_time = time.strftime('%Y-%m-%d %H:%M:%S')  # 상태 변경 시간 업데이트

            with lock:
                # 감지된 이미지를 저장
                cv2.imwrite(img_path, frame)

        time.sleep(0.1)  # n초마다 이미지 캡처

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
            # 이미지 파일을 바이너리로 읽어서 전송
            with open(img_path, 'rb') as f:
                image_data = f.read()
            return Response(image_data, mimetype='image/jpeg')
        else:
            return jsonify({'status': 'error', 'message': 'No image available.'})

@app.route('/status', methods=['GET'])
def get_detection_status():
    return jsonify({'detected': detected})

@app.route('/status_with_time', methods=['GET'])
def get_detection_status_with_time():
    return jsonify({'detected': detected, 'last_update': last_detection_time})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)