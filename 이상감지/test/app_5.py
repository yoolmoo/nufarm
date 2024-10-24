from flask import Flask, jsonify, render_template, Response, request
import cv2
import threading
import os
import time
from ultralytics import YOLO
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

model1 = YOLO('abnormal.pt')
model2 = YOLO('growth.pt')

# 웹캠 인덱스
camera = cv2.VideoCapture(2)

# 이미지 저장 경로
img_path1 = './static/captured_image_model1.jpg'
img_path2 = './static/captured_image_model2.jpg'

# 파일 접근을 보호하기 위한 Lock 생성
lock = threading.Lock()

# 클래스 이름과 색상 정의 (각 모델에 대해)
class_names_model1 = {
    0: 'hole',
    1: 'wither'
}
class_names_model2 = {
    0: 'level_1',
    1: 'level_2',
    2: 'level_3',
}

class_colors_model1 = {
    0: (255, 0, 0),  # 빨간색 (구멍)
    1: (127, 255, 212),  # 민트색 (시든거)
}

class_colors_model2 = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 255),
}

# 감지 상태를 저장하는 변수
status = False

# # 백엔드로 이상감지 정보 전달
# def send_notification(class_counts):
#     url = "http://3.34.153.235:8080//api/notification/save"
#     params = {
#         'hole': class_counts.get('hole', 0),
#         'wither': class_counts.get('wither', 0)
#     }

#     print(f"Sending request to {url} with params: {params}")

#     # 403 오류 발생 : API 호출 시 인증 정보가 누락되었거나 잘못된 경우. 예를 들어, API 키, 인증 토큰이 없거나 잘못된 값을 사용할 때
#     try:
#         response = requests.get(url, params=params)
#         if response.status_code == 200:
#             print("Notification sent successfully.")
#         else:
#             print(f"Failed to send notification: {response.status_code}")
#     except Exception as e:
#         print(f"Error sending notification: {e}")

def capture_image_periodically(model, img_path, class_names, class_colors):
    global status, reset_status_flag
    while True:
        camera.grab()
        ret, frame = camera.retrieve()

        if ret:
            results = model(frame)
            detected_counts = {name: 0 for name in class_names.values()}

            # 감지된 객체 처리
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    label_id = int(box.cls)
                    color = class_colors.get(label_id, (255, 255, 255))
                    label = class_names.get(label_id, 'Unknown')

                    # 감지된 클래스별 개수 증가
                    if label in detected_counts:
                        detected_counts[label] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # reset_status_flag가 True일 때 일정 시간 동안 상태를 변경하지 않음
            if reset_status_flag:
                print("Reset status flag is active, not changing status.")
                time.sleep(100)  # 5초 동안 상태 변경을 막음 (필요에 따라 조정 가능)
                reset_status_flag = False

            # 감지된 객체가 있는 경우 상태를 True로 변경
            elif any(count > 0 for count in detected_counts.values()):
                with status_lock:
                    status = True
                    print("Status changed to True due to new detection.")

            # 감지된 객체가 없는 경우 상태를 False로 유지
            else:
                with status_lock:
                    status = False

            # 로그로 상태 출력
            print(f'Status: {status}, Counts: {detected_counts}')

            with lock:
                cv2.imwrite(img_path, frame)

        time.sleep(2)

# def capture_image_periodically(model, img_path, class_names, class_colors):
#     global status
#     while True:
#         camera.grab()
#         ret, frame = camera.retrieve()

#         if ret:
#             results = model(frame)
#             detected_counts = {name: 0 for name in class_names.values()}

            
#             for result in results:
#                 boxes = result.boxes
#                 for box in boxes:
#                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#                     label_id = int(box.cls)
#                     color = class_colors.get(label_id, (255, 255, 255))
#                     label = class_names.get(label_id, 'Unknown')

#                     # 감지된 클래스별 개수 증가
#                     if label in detected_counts:
#                         detected_counts[label] += 1

#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(frame, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
#             if all(count == 0 for count in detected_counts.values()):
#                 status = False
#             else:
#                 status = True

#             # 로그로 상태 출력
#             print(f'Status: {status}, Counts: {detected_counts}')


#             with lock:
#                 cv2.imwrite(img_path, frame)

#         #time.sleep(0.1)
#         time.sleep(2)


# 스레드 생성 (model1에 대한 감지)
thread1 = threading.Thread(target=capture_image_periodically, args=(model1, img_path1, class_names_model1, class_colors_model1))
thread1.daemon = True
thread1.start()

# 스레드 생성 (model2에 대한 감지)
thread2 = threading.Thread(target=capture_image_periodically, args=(model2, img_path2, class_names_model2, class_colors_model2))
thread2.daemon = True
thread2.start()

@app.route('/model1')
def index_model1():
    return render_template('index.html')

@app.route('/model2')
def index_model2():
    return render_template('index.html')

@app.route('/image_model1', methods=['GET'])
def get_image_model1():
    with lock:
        if os.path.exists(img_path1):
            with open(img_path1, 'rb') as f:
                image_data = f.read()
            return Response(image_data, mimetype='image/jpeg')
        else:
            return jsonify({'status': 'error', 'message': 'No image available for model1.'})

@app.route('/image_model2', methods=['GET'])
def get_image_model2():
    with lock:
        if os.path.exists(img_path2):
            with open(img_path2, 'rb') as f:
                image_data = f.read()
            return Response(image_data, mimetype='image/jpeg')
        else:
            return jsonify({'status': 'error', 'message': 'No image available for model2.'})

# 1) 유림 >> 건우 : model1에서 감지된 박스가 있을 때 status를 true로 보내는 API
@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({'status': status})

# 전역 변수 추가
reset_status_flag = False
status_lock = threading.Lock()

# 2) 건우 >> 유림 : 버튼 클릭 시 status를 false로 바꾸는 API
@app.route('/reset_status', methods=['POST'])
def reset_status():
    global status, reset_status_flag
    with status_lock:
        status = False
        reset_status_flag = True  # 플래그 설정
    print(f'Status = {status}')
    return jsonify({'status': status})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
