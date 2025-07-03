from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import os
import numpy as np
from io import BytesIO
import json
import cv2
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "silent_face_anti_spoofing"))
from silent_face_anti_spoofing.test import test

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

EMBEDDING_FILE = 'face_embeddings.json'  # File lưu embedding

# Constants
FACE_DETECTION_THRESHOLD = 0.6
FACE_MATCHING_THRESHOLD = 0.4
MIN_FACE_SIZE = 100

def process_image(image_data):
    """Convert image data to numpy array"""
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def detect_face(image):
    """Detect face in image and return face encoding"""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_image)

    if not face_locations:
        return None, "Không phát hiện khuôn mặt trong ảnh"

    if len(face_locations) > 1:
        return None, "Phát hiện nhiều khuôn mặt. Vui lòng chỉ chụp một khuôn mặt"

    # Get face encoding
    face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]

    # Check face size
    top, right, bottom, left = face_locations[0]
    face_size = max(bottom - top, right - left)
    if face_size < MIN_FACE_SIZE:
        return None, "Khuôn mặt quá nhỏ. Vui lòng đứng gần camera hơn"

    return face_encoding, None

# Đăng ký khuôn mặt mới
# @app.route('/register-face', methods=['POST'])
# def register_face():
#     if 'image' not in request.files or 'employeeCode' not in request.form:
#         return jsonify({"success": False, "message": "Missing image or employeeCode"}), 400
#     file = request.files['image']
#     employee_code = request.form['employeeCode']
#     if file.filename == '':
#         return jsonify({"success": False, "message": "No selected file"}), 400
#     # Tạo folder theo mã nhân viên
#     save_dir = os.path.join(UPLOAD_FOLDER, secure_filename(employee_code))
#     os.makedirs(save_dir, exist_ok=True)
#     # Lưu file ảnh
#     filename = secure_filename(file.filename)
#     save_path = os.path.join(save_dir, filename)
#     file.save(save_path)
#     # Trích xuất embedding
#     img = face_recognition.load_image_file(save_path)
#     encodings = face_recognition.face_encodings(img)
#     if len(encodings) == 0:
#         return jsonify({"success": False, "message": "No face detected in the image"}), 400
#     embedding = encodings[0].tolist()
#     # Đọc file embedding hiện tại (nếu có)
#     data = {}
#     if os.path.exists(EMBEDDING_FILE):
#         try:
#             with open(EMBEDDING_FILE, 'r') as f:
#                 data = json.load(f)
#             if not isinstance(data, dict):
#                 data = {}
#         except Exception as e:
#             print(f"Error reading embedding file: {e}")
#             data = {}
#     # Thêm hoặc cập nhật embedding cho mã nhân viên
#     is_update = employee_code in data
#     data[employee_code] = embedding
#     with open(EMBEDDING_FILE, 'w') as f:
#         json.dump(data, f)
#     if is_update:
#         return jsonify({"success": True, "message": "Face updated successfully"})
#     else:
#         return jsonify({"success": True, "message": "Face registered successfully"})

# Nhận diện khuôn mặt
@app.route('/compare-image', methods=['POST', 'OPTIONS'])
def compare_image():
    if request.method == 'OPTIONS':
        return '', 200
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image file provided"}), 400
    file = request.files['image']
    file_bytes = file.read()
    # Trích xuất embedding từ ảnh mới
    img = face_recognition.load_image_file(BytesIO(file_bytes))
    encodings = face_recognition.face_encodings(img)
    if len(encodings) == 0:
        return jsonify({"success": False, "message": "Bạn không có trong hệ thống công ty"}), 400
    if len(encodings) > 1:
        return jsonify({"success": False, "message": "More than one face detected in the image. Please make sure only one face is visible."}), 400
    new_embedding = encodings[0]
    # Load embedding đã lưu
    if not os.path.exists(EMBEDDING_FILE):
        return jsonify({"success": False, "message": "No registered embeddings found"}), 404
    with open(EMBEDDING_FILE, 'r') as f:
        data = json.load(f)
    if not data:
        return jsonify({"success": False, "message": "No registered embeddings found"}), 404
    employee_codes = list(data.keys())
    embeddings = np.array([data[code] for code in employee_codes])
    # So sánh embedding
    distances = np.linalg.norm(embeddings - new_embedding, axis=1)
    min_idx = np.argmin(distances)
    threshold = 0.4  # Giảm ngưỡng nhận diện để tăng độ chính xác
    if distances[min_idx] < threshold:
        return jsonify({
            "success": True,
            "employeeCode": employee_codes[min_idx],
        })
    else:
        return jsonify({
            "success": False,
            "message": "No matching face found"
        })

@app.route('/extract-embedding', methods=['POST'])
def extract_embedding():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image file provided"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400
    img = face_recognition.load_image_file(file)
    encodings = face_recognition.face_encodings(img)
    if len(encodings) == 0:
        return jsonify({"success": False, "message": "No face detected in the image"}), 400
    embedding = encodings[0].tolist()
    return jsonify({"success": True, "faceDescriptor": embedding})

@app.route('/process-face', methods=['POST'])
def process_face():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Không tìm thấy ảnh trong request'
            }), 400

        image_file = request.files['image']
        image_data = image_file.read()

        # Process image
        image = process_image(image_data)
        face_encoding, error = detect_face(image)

        if error:
            return jsonify({
                'success': False,
                'message': error
            }), 400

        return jsonify({
            'success': True,
            'faceDescriptor': face_encoding.tolist()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi xử lý ảnh: {str(e)}'
        }), 500

@app.route('/compare-faces', methods=['POST'])
def compare_faces():
    try:
        data = request.json
        face1 = np.array(data['face1'])
        face2 = np.array(data['face2'])

        # Calculate face distance
        face_distance = face_recognition.face_distance([face1], face2)[0]
        is_match = face_distance < FACE_MATCHING_THRESHOLD

        return jsonify({
            'success': True,
            'isMatch': bool(is_match),
            'distance': float(face_distance)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi so sánh khuôn mặt: {str(e)}'
        }), 500

@app.route('/find-matching-user', methods=['POST'])
def find_matching_user():
    try:
        data = request.json
        face_descriptor = np.array(data['faceDescriptor'])
        users = data['users']

        best_match = None
        highest_similarity = 0

        for user in users:
            if not user.get('faceDescriptors'):
                continue

            for stored_descriptor in user['faceDescriptors']:
                stored_face = np.array(stored_descriptor)
                face_distance = face_recognition.face_distance([face_descriptor], stored_face)[0]
                similarity = 1 - face_distance

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = user['id']

        if not best_match or highest_similarity < FACE_MATCHING_THRESHOLD:
            return jsonify({
                'success': True,
                'match': False
            })

        return jsonify({
            'success': True,
            'match': True,
            'userId': best_match,
            'similarity': float(highest_similarity)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi tìm kiếm người dùng: {str(e)}'
        }), 500

@app.route('/check-real-face', methods=['POST'])
def check_real_face():
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Không tìm thấy ảnh trong request'
            }), 400

        file = request.files['image']
        image_stream = file.read()

        # Convert binary to numpy array
        file_bytes = np.frombuffer(image_stream, np.uint8)

        # Decode image (numpy array) từ bytes
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Resize image to 4:3 ratio
        # new_image = resize_to_4_3(image)

        label = test(
                image=image,
                model_dir='./silent_face_anti_spoofing/resources/anti_spoof_models',
                device_id=0,
                )

        if label == 1:
            return jsonify({
                'success': True,
                'isReal': True,
                'message': 'Khuôn mặt hợp lệ'
            }), 200
        else:
            return jsonify({
                'success': True,
                'isReal': False,
                'message': 'Phát hiện khuôn mặt giả'
            }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi xử lý ảnh: {str(e)}'
        }), 500


@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    print('Exception:', e)
    print(traceback.format_exc())
    return jsonify({"success": False, "message": str(e)}), 500

port = 9000
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, ssl_context=('./SSL/localhost+1.pem','./SSL/localhost+1-key.pem'))
