# kakao_api.py

import requests
import json
import os
import logging
import time

logger = logging.getLogger(__name__)

from config import KAKAO_REST_API_KEY, KAKAO_REDIRECT_URI, KAKAO_TOKEN_FILE

# ==============================================================================
# 카카오톡 API 함수들 (텍스트 및 이미지 메시지 전송)
# ==============================================================================

TEXT_SEND_URL = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
IMAGE_UPLOAD_URL = "https://kapi.kakao.com/v1/api/talk/message/image/upload"


def get_kakao_auth_url():
    """
    카카오톡 인가 코드를 요청할 URL을 생성하여 반환합니다.
    """
    auth_url = (
        f"https://kauth.kakao.com/oauth/authorize?response_type=code"
        f"&client_id={KAKAO_REST_API_KEY}&redirect_uri={KAKAO_REDIRECT_URI}"
        f"&scope=talk_message"
    )
    return auth_url


def get_kakao_tokens(auth_code):
    """
    인가 코드를 사용하여 Access Token과 Refresh Token을 발급받습니다.
    """
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": KAKAO_REST_API_KEY,
        "redirect_uri": KAKAO_REDIRECT_URI,
        "code": auth_code,
    }
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        tokens = response.json()
        if "access_token" in tokens:
            logger.info("카카오톡 토큰 발급 성공!")
            save_tokens(tokens)
            return tokens
        else:
            logger.error(f"카카오톡 토큰 발급 실패 (응답에 access_token 없음): {tokens}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"카카오톡 토큰 발급 중 네트워크 오류: {e}")
        return None
    except json.JSONDecodeError:
        logger.error(f"카카오톡 토큰 응답 JSON 디코딩 실패: {response.text}")
        return None


def refresh_kakao_token(refresh_token):
    """
    Refresh Token을 사용하여 Access Token을 갱신합니다.
    """
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": KAKAO_REST_API_KEY,
        "refresh_token": refresh_token,
    }
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        tokens = response.json()
        if "access_token" in tokens:
            logger.info("카카오톡 토큰 갱신 성공!")
            current_tokens = load_tokens()
            if current_tokens:
                current_tokens['access_token'] = tokens['access_token']
                if 'refresh_token' in tokens:
                    current_tokens['refresh_token'] = tokens['refresh_token']
                save_tokens(current_tokens)
                return current_tokens
            else:
                logger.error("기존 토큰을 로드할 수 없어 갱신된 토큰을 저장할 수 없습니다.")
                return None
        else:
            logger.error(f"카카오톡 토큰 갱신 실패 (응답에 access_token 없음): {tokens}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"카카오톡 토큰 갱신 중 네트워크 오류: {e}")
        return None
    except json.JSONDecodeError:
        logger.error(f"카카오톡 토큰 갱신 응답 JSON 디코딩 실패: {response.text}")
        return None


def save_tokens(tokens):
    """
    발급받은 토큰을 파일에 저장합니다.
    """
    try:
        with open(KAKAO_TOKEN_FILE, "w", encoding='utf-8') as f:
            json.dump(tokens, f, indent=2, ensure_ascii=False)
        logger.info(f"카카오톡 토큰이 {KAKAO_TOKEN_FILE}에 저장되었습니다.")
    except IOError as e:
        logger.error(f"토큰 파일을 저장할 수 없습니다: {e}")


def load_tokens():
    """
    저장된 토큰을 파일에서 불러옵니다.
    """
    if os.path.exists(KAKAO_TOKEN_FILE):
        try:
            with open(KAKAO_TOKEN_FILE, "r", encoding='utf-8') as f:
                tokens = json.load(f)
            logger.info(f"카카오톡 토큰이 {KAKAO_TOKEN_FILE}에서 로드되었습니다.")
            return tokens
        except json.JSONDecodeError:
            logger.error(f"토큰 파일 {KAKAO_TOKEN_FILE}이 유효한 JSON 형식이 아닙니다.")
            return None
        except IOError as e:
            logger.error(f"토큰 파일을 읽을 수 없습니다: {e}")
            return None
    logger.info(f"토큰 파일 {KAKAO_TOKEN_FILE}이 존재하지 않습니다.")
    return None


def send_kakao_text_message(access_token, message_text):
    """
    카카오톡 나에게 텍스트 메시지를 전송합니다.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    template_object = {
        "object_type": "text",
        "text": message_text,
        "link": {
            "web_url": "https://developers.kakao.com",
            "mobile_web_url": "https://developers.kakao.com"
        },
        "button_title": "자세히 보기"
    }
    data = {"template_object": json.dumps(template_object)}
    try:
        response = requests.post(TEXT_SEND_URL, headers=headers, data=data)
        response.raise_for_status()
        result = response.json()
        if result.get("result_code") == 0:
            logger.info("카카오톡 텍스트 메시지 전송 성공!")
            return True
        else:
            logger.error(f"카카오톡 텍스트 메시지 전송 실패: {result}")
            if result.get("code") == -401:
                logger.warning("Access Token이 만료되었거나 유효하지 않습니다.")
                return 'token_expired'
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"카카오톡 텍스트 메시지 전송 중 네트워크 오류: {e}")
        return False
    except json.JSONDecodeError:
        logger.error(f"카카오톡 텍스트 메시지 전송 응답 JSON 디코딩 실패: {response.text}")
        return False


def send_kakao_image_message(access_token, image_path, message_text="넘어짐 감지!"):
    """
    나에게 이미지 전송을 시도하지만, 메모 API는 이미지 업로드를 지원하지 않으므로
    항상 텍스트로 대체 전송하도록 합니다.
    """
    logger.info("메모 방식이므로 이미지 전송이 불가능하여 텍스트로 전송합니다.")
    return send_kakao_text_message(access_token, message_text)


def initialize_kakao_tokens(auth_code=None):
    """
    저장된 토큰을 로드하거나, Refresh Token으로 갱신하거나,
    새로운 인가 코드로 토큰을 발급받아 반환합니다.
    """
    tokens = load_tokens()
    if tokens and 'refresh_token' in tokens:
        logger.info("Refresh Token을 사용하여 Access Token 갱신 시도...")
        new_tokens = refresh_kakao_token(tokens['refresh_token'])
        if new_tokens:
            return new_tokens
        else:
            logger.warning("Refresh Token 갱신 실패. 새로운 인증이 필요합니다.")
            return None
    if auth_code:
        return get_kakao_tokens(auth_code)
    logger.info("카카오톡 토큰이 없거나 유효하지 않아 새로운 인증이 필요합니다.")
    return None


def initialize_and_send_text(auth_code=None, message_text="테스트 메시지"):
    """
    인증 코드 또는 기존 토큰을 초기화한 뒤, 메시지를 전송합니다.
    """
    tokens = initialize_kakao_tokens(auth_code)
    if not tokens:
        logger.error("토큰이 없거나 초기화 실패")
        return False
    result = send_kakao_text_message(tokens['access_token'], message_text)
    if result == 'token_expired':
        tokens = refresh_kakao_token(tokens['refresh_token'])
        if not tokens:
            return False
        return send_kakao_text_message(tokens['access_token'], message_text) is True
    return result


# 모듈 직접 실행 테스트
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("=== 카카오톡 메시지 테스트 ===")
    auth_code = input("인가코드 입력 (처음 실행 시 필요, 이미 발급된 경우 그냥 Enter): ").strip() or None
    # 텍스트 전송 예
    success_text = initialize_and_send_text(auth_code, "텍스트 메시지 테스트입니다.")
    print("텍스트 전송 결과:", success_text)
    # 이미지 전송 예 (항상 텍스트로 대체)
    success_img = send_kakao_image_message(tokens=load_tokens()['access_token'], image_path="test.jpg", message_text="이미지 테스트")
    print("이미지 전송(텍스트 대체) 결과:", success_img)
