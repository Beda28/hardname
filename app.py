# app.py

from flask import Flask, render_template, Response, redirect, url_for, request, jsonify
import cv2
import torch
import numpy as np # MediaPipe 제거했으므로 불필요할 수 있으나, 혹시 몰라 유지.
import time
import os
from datetime import datetime
import threading
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 사용자 정의 모듈 임포트
from config import (
    CAMERA_INDEX, YOLO_MODEL_NAME, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    FALL_DIFFERENCE_THRESHOLD, MIN_PERSON_HEIGHT_PIXELS, FALL_COOLDOWN_SECONDS,
    KAKAO_REST_API_KEY, KAKAO_REDIRECT_URI, KAKAO_TOKEN_FILE, WEB_SERVER_PORT,
    FALL_MIN_CONSECUTIVE_FRAMES
)
from kakao_api import (
    initialize_kakao_tokens, send_kakao_image_message,
    get_kakao_auth_url, load_tokens, refresh_kakao_token
)

app = Flask(__name__)

# ==============================================================================
# 전역 변수 및 스레드 설정
# ==============================================================================
camera = None
yolo_model = None
latest_frame = None
frame_lock = threading.Lock()
detection_thread = None
running = False
kakao_access_token = None
kakao_auth_status = "인증 필요"
fall_status = "안전" # "넘어짐 감지!" 또는 "안전"
last_alert_time = 0  # 넘어짐 알림용 타임스탬프

consecutive_fall_frames = 0 # 넘어짐 조건 충족 연속 프레임 카운터

# ==============================================================================
# 초기 설정 함수
# ==============================================================================
def initialize_system():
    global yolo_model, camera, kakao_access_token, kakao_auth_status

    logger.info("시스템 초기화 중...")

    logger.info(f"YOLOv5 모델 로딩 중: {YOLO_MODEL_NAME}...")
    try:
        yolo_model = torch.hub.load('ultralytics/yolov5', YOLO_MODEL_NAME, pretrained=True, trust_repo=True)
        yolo_model.conf = CONFIDENCE_THRESHOLD
        yolo_model.iou = IOU_THRESHOLD
        logger.info("YOLOv5 모델 로딩 완료.")
    except Exception as e:
        logger.error(f"YOLOv5 모델 로딩 실패: {e}")
        logger.error("인터넷 연결을 확인하고 'pip install ultralytics'가 완료되었는지 확인하세요.")
        yolo_model = None
        return False

    logger.info(f"노트북 카메라 연결 시도 중 (인덱스: {CAMERA_INDEX})...")
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        logger.error(f"카메라 (인덱스: {CAMERA_INDEX})를 열 수 없습니다.")
        logger.error("카메라가 제대로 연결되어 있고, 다른 프로그램에서 사용 중이 아닌지 확인하세요.")
        camera = None
        return False
    logger.info("카메라 연결 성공.")

    tokens = initialize_kakao_tokens()
    if tokens:
        kakao_access_token = tokens['access_token']
        kakao_auth_status = "인증 완료"
        logger.info("카카오톡 토큰 로드/갱신 성공.")
    else:
        kakao_auth_status = "카카오 인증 필요"
        logger.warning("카카오톡 인증이 필요합니다. 웹 페이지에서 인증을 진행해주세요.")

    return True

# ==============================================================================
# 넘어짐 감지 스레드 (영상 알고리즘 적용)
# ==============================================================================
def detect_falls_and_stream():
    global latest_frame, kakao_access_token, last_alert_time, fall_status, running, kakao_auth_status
    global consecutive_fall_frames

    logger.info("감지 스레드 시작.")
    running = True

    while running:
        if camera is None or not camera.isOpened():
            logger.error("카메라가 연결되지 않았거나 열리지 않았습니다. 스레드를 중지합니다.")
            running = False
            break
        if yolo_model is None:
            logger.error("YOLO 모델이 로드되지 않았습니다. 스레드를 중지합니다.")
            running = False
            break

        ret, frame = camera.read()
        if not ret:
            logger.error("프레임을 읽어올 수 없습니다. 카메라 스트림 종료.")
            running = False
            break

        # YOLOv5로 객체 감지
        yolo_results = yolo_model(frame)
        detections = yolo_results.pandas().xyxy[0] # 감지된 객체들의 정보

        person_detected_in_this_frame = False
        current_frame_is_fall_candidate = False # 현재 프레임에서 넘어짐 조건이 일시적으로 만족했는지

        # 이 프레임에서 최종적으로 "넘어짐 감지!" 상태인지
        fall_detected_in_frame = False

        for i in range(len(detections)):
            label = detections.loc[i, 'name']
            confidence = detections.loc[i, 'confidence']
            xmin, ymin, xmax, ymax = detections.loc[i, ['xmin', 'ymin', 'xmax', 'ymax']].astype(int)

            if label == 'person':
                person_detected_in_this_frame = True
                width = xmax - xmin
                height = ymax - ymin

                if height < MIN_PERSON_HEIGHT_PIXELS:
                    # 너무 작게 검출된 객체는 무시
                    continue

                # ==========================================================
                # 영상에서 설명된 넘어짐 감지 알고리즘 적용
                # "height - width" 임계값 기반
                # ==========================================================
                threshold_value = height - width

                # 신뢰도가 충분하고, 높이-폭 임계값을 만족하면 넘어짐 후보
                if confidence > CONFIDENCE_THRESHOLD and threshold_value < FALL_DIFFERENCE_THRESHOLD:
                    current_frame_is_fall_candidate = True # 이 프레임에서 넘어짐 조건 만족

                # 시각화: 바운딩박스 및 텍스트
                color = (0, 255, 0) # 기본 초록색 (안전)
                display_text = f'{label} {confidence:.2f}'

                # 넘어짐 후보 조건이 만족하면 색상 및 텍스트 변경 (최종 감지 여부와 무관)
                if current_frame_is_fall_candidate:
                    color = (0, 165, 255) # 오렌지색 (넘어짐 후보)
                    display_text = f'Candidate ({threshold_value})'

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, display_text, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 한 프레임에 여러 사람이 감지될 수 있으므로, 한 명이라도 넘어짐 후보면 카운트
                # 하지만, 이 알고리즘은 보통 "한 사람"에 대해 적용하는 것이 일반적
                # 여기서는 가장 먼저 감지된 사람에 대해 넘어짐 로직을 적용하는 것으로 간주.
                # 더 복잡한 시나리오(여러 사람 중 누가 넘어졌는지)는 MediaPipe Pose가 더 적합.
                break # 첫 번째 사람만 처리하고 루프 종료 (간단한 구현을 위해)

            else: # 사람이 아닌 다른 객체는 일반 초록색으로 표시
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # 연속 프레임 카운터 업데이트
        if person_detected_in_this_frame and current_frame_is_fall_candidate:
            consecutive_fall_frames += 1
        else:
            consecutive_fall_frames = 0 # 조건 불만족 또는 사람 미감지 시 초기화

        # 최종 넘어짐 감지 판정
        if consecutive_fall_frames >= FALL_MIN_CONSECUTIVE_FRAMES:
            fall_detected_in_frame = True
            fall_status = "넘어짐 감지!" # 웹 페이지 상태 업데이트
        else:
            fall_status = "안전"


        # 넘어짐 상태 발생 시 카카오톡 알림
        if fall_detected_in_frame:
            current_time = time.time()
            # 쿨다운이 지난 경우에만 알림 발송
            if kakao_access_token and (current_time - last_alert_time) > FALL_COOLDOWN_SECONDS:
                logger.info(f"넘어짐 감지! (연속 {consecutive_fall_frames} 프레임) 알림 전송 시도...")
                message_text = "🚨 넘어짐 감지! 즉시 확인하세요."

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"alert_{timestamp}.jpg"
                cv2.imwrite(image_filename, frame) # 현재 프레임을 이미지로 저장

                send_status = send_kakao_image_message(kakao_access_token, image_filename, message_text)
                if send_status == 'token_expired':
                    logger.warning("카카오톡 Access Token 만료. Refresh Token으로 갱신 시도...")
                    tokens = load_tokens()
                    if tokens and 'refresh_token' in tokens:
                        new_tokens = refresh_kakao_token(tokens['refresh_token'])
                        if new_tokens:
                            kakao_access_token = new_tokens['access_token']
                            kakao_auth_status = "인증 완료 (토큰 갱신)"
                            logger.info("토큰 갱신 성공. 재전송 시도...")
                            second_try = send_kakao_image_message(kakao_access_token, image_filename, message_text)
                            if second_try:
                                logger.info("카카오톡 메시지 재전송 완료.")
                                last_alert_time = current_time # 재전송 성공 시 쿨다운 업데이트
                            else:
                                logger.error("카카오톡 메시지 재전송 실패.")
                        else:
                            kakao_auth_status = "카카오 인증 필요 (갱신 실패)"
                            logger.error("토큰 갱신 실패. 수동 인증이 필요합니다.")
                    else:
                        kakao_auth_status = "카카오 인증 필요 (Refresh Token 없음)"
                        logger.error("Refresh Token 없음. 수동 인증이 필요합니다.")
                elif send_status:
                    logger.info("카카오톡 메시지 전송 완료.")
                    last_alert_time = current_time # 전송 성공 시 쿨다운 업데이트
                else:
                    logger.error("카카오톡 메시지 전송 실패.")

                # 임시 이미지 파일 삭제
                if os.path.exists(image_filename):
                    os.remove(image_filename)
                    logger.info(f"임시 이미지 파일 {image_filename} 삭제됨.")

        # 최종 처리된 프레임을 저장하여 웹 스트림으로 전송
        with frame_lock:
            latest_frame = frame.copy()

        # 프레임 처리 속도 조절 (필요시)
        # time.sleep(0.01) # 너무 느리면 주석 처리하거나 값을 줄이세요.

    logger.info("감지 스레드 종료.")
    if camera:
        camera.release()
        logger.info("카메라 리소스 해제 완료 (스레드 내부).")

# ==============================================================================
# Flask 웹 서버 라우트 (변경 없음)
# ==============================================================================
@app.route('/')
def index():
    global kakao_auth_status
    auth_url = get_kakao_auth_url()
    if not os.path.exists(KAKAO_TOKEN_FILE):
        kakao_auth_status = "카카오 인증 필요"

    return render_template('index.html',
                           kakao_auth_status=kakao_auth_status,
                           kakao_auth_url=auth_url)

@app.route('/oauth')
def oauth():
    global kakao_access_token, kakao_auth_status
    code = request.args.get('code')
    if code:
        tokens = initialize_kakao_tokens(auth_code=code)
        if tokens:
            kakao_access_token = tokens['access_token']
            kakao_auth_status = "인증 완료"
            logger.info("카카오톡 인가 코드 수신 및 토큰 발급 완료.")
            return redirect(url_for('index'))
        else:
            kakao_auth_status = "카카오 인증 실패"
            logger.error("카카오 인증 실패. 콘솔을 확인하세요.")
            return "카카오 인증 실패. 콘솔을 확인하세요.", 500
    else:
        error = request.args.get('error')
        error_description = request.args.get('error_description')
        kakao_auth_status = f"카카오 인증 취소 또는 오류: {error} ({error_description})"
        logger.error(f"카카오 인증 취소 또는 오류: {error_description}")
        return f"카카오 인증 취소 또는 오류: {error_description}", 400

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global latest_frame
        while running:
            with frame_lock:
                if latest_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', latest_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not ret:
                        logger.warning("프레임 JPEG 인코딩 실패.")
                        continue
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return jsonify({
        'kakao_auth_status': kakao_auth_status,
        'fall_status': fall_status,
        'running': running
    })

# ==============================================================================
# 앱 종료 시 처리
# ==============================================================================
@app.teardown_appcontext
def shutdown_detection_thread_on_teardown(exception=None):
    global running
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        logger.info("애플리케이션 종료 요청 감지. 감지 스레드 종료 시도...")
        running = False
        if detection_thread and detection_thread.is_alive():
            detection_thread.join(timeout=5)
            if detection_thread.is_alive():
                logger.warning("감지 스레드가 5초 내에 종료되지 않았습니다. 강제 종료될 수 있습니다.")
            else:
                logger.info("감지 스레드 종료 완료.")
        else:
            logger.info("감지 스레드가 실행 중이 아니거나 이미 종료되었습니다.")

# ==============================================================================
# Flask 앱 실행
# ==============================================================================
if __name__ == '__main__':
    logger.info("시스템 초기화 및 감지 스레드 시작 전...")
    if initialize_system():
        detection_thread = threading.Thread(target=detect_falls_and_stream)
        detection_thread.daemon = True
        detection_thread.start()
        logger.info("감지 스레드 시작됨.")
    else:
        logger.error("시스템 초기화 실패. 웹 서버가 제대로 작동하지 않을 수 있습니다.")

    logger.info(f"웹 서버 시작 중: http://localhost:{WEB_SERVER_PORT}")
    app.run(host='0.0.0.0', port=WEB_SERVER_PORT, debug=False, threaded=True)