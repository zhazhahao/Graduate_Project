import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, url_for, send_file, Response
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from models.resnet_lgca import ResNet50_LGCA
import torch.nn.functional as F
from utils.visualization import AttentionVisualizer
import matplotlib.pyplot as plt
import matplotlib
import cv2
import datetime
from pathlib import Path
import threading
import queue
import time
matplotlib.use('Agg')  # 设置后端为Agg
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import io
import base64
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # 忽略UserWarning

# 创建Flask应用，指定模板目录
app_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(app_dir, 'app', 'templates')
static_dir = os.path.join(app_dir, 'app', 'static')
app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)

# 配置文件夹
UPLOAD_FOLDER = os.path.join(static_dir, 'uploads')
GALLERY_FOLDER = os.path.join(static_dir, 'gallery')
DETECTED_FOLDER = os.path.join(static_dir, 'detected_persons')  # 新增检测人物保存目录
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}  # 添加视频格式
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GALLERY_FOLDER'] = GALLERY_FOLDER
app.config['DETECTED_FOLDER'] = DETECTED_FOLDER

# 确保必要的目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GALLERY_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

# 加载行人检测模型
person_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# 视频处理队列
video_queue = queue.Queue()
processing_thread = None
is_processing = False

def process_video_frame(frame):
    """处理单帧视频并检测人物"""
    # 转换为灰度图进行检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 检测人物
    persons = person_detector.detectMultiScale(gray, 1.1, 4)
    
    detected_persons = []
    for (x, y, w, h) in persons:
        # 扩大检测框以包含完整人物
        y = max(0, y - h//10)
        h = min(frame.shape[0] - y, int(h * 1.2))
        
        person_img = frame[y:y+h, x:x+w]
        if person_img.size > 0:
            # 转换为PIL图像
            person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
            detected_persons.append({
                'image': person_pil,
                'bbox': (x, y, w, h)
            })
    
    return detected_persons

def save_detected_person(person_img, similarity=None):
    """保存检测到的人物图片"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f'person_{timestamp}.jpg'
    save_path = os.path.join(app.config['DETECTED_FOLDER'], filename)
    person_img.save(save_path)
    return filename

def process_video_stream():
    """处理视频流的主函数"""
    global is_processing
    
    while is_processing:
        if not video_queue.empty():
            frame = video_queue.get()
            if frame is None:
                continue
                
            # 检测人物
            detected_persons = process_video_frame(frame)
            
            for person in detected_persons:
                try:
                    # 提取特征
                    person_tensor = transform(person['image'])
                    person_tensor = person_tensor.unsqueeze(0).to(device)
                    with torch.no_grad():
                        _, features = model(person_tensor)
                    
                    # 与库中的图片比较
                    max_similarity = 0
                    match_found = False
                    
                    for gallery_file in os.listdir(app.config['GALLERY_FOLDER']):
                        if allowed_file(gallery_file):
                            gallery_path = os.path.join(app.config['GALLERY_FOLDER'], gallery_file)
                            gallery_features = extract_features(gallery_path)
                            similarity = compute_similarity(features.cpu().numpy(), gallery_features)
                            
                            if similarity > max_similarity:
                                max_similarity = similarity
                            
                            if similarity >= 0.75:  # 相似度阈值
                                match_found = True
                                break
                    
                    # 如果是新的人物，保存到检测文件夹
                    if not match_found:
                        save_detected_person(person['image'])
                
                except Exception as e:
                    print(f"处理检测到的人物时出错: {str(e)}")
                    continue
        
        time.sleep(0.1)  # 避免CPU占用过高

@app.route('/start_video', methods=['POST'])
def start_video():
    """开始视频处理"""
    global processing_thread, is_processing
    
    if 'video' not in request.files:
        return jsonify({'error': '没有上传视频文件'})
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    if not video_file.filename.lower().endswith(('.mp4', '.avi')):
        return jsonify({'error': '不支持的视频格式'})
    
    # 保存视频文件
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)
    
    # 启动视频处理线程
    is_processing = True
    processing_thread = threading.Thread(target=process_video)
    processing_thread.start()
    
    return jsonify({'success': True, 'message': '视频处理已开始'})

@app.route('/stop_video', methods=['POST'])
def stop_video():
    """停止视频处理"""
    global is_processing
    is_processing = False
    if processing_thread:
        processing_thread.join()
    return jsonify({'success': True, 'message': '视频处理已停止'})

def process_video():
    """视频处理主函数"""
    cap = cv2.VideoCapture(0)  # 使用默认摄像头，也可以传入视频文件路径
    
    while is_processing:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 将帧添加到处理队列
        if video_queue.qsize() < 10:  # 限制队列大小
            video_queue.put(frame)
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    """返回处理后的视频流"""
    def generate():
        while True:
            if not video_queue.empty():
                frame = video_queue.get()
                if frame is None:
                    continue
                    
                # 在帧上绘制检测框
                detected_persons = process_video_frame(frame)
                for person in detected_persons:
                    x, y, w, h = person['bbox']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 将帧转换为JPEG格式
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            
            time.sleep(0.1)
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet50_LGCA(num_classes=1501, use_lgca=True)
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'model_best.pth')
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_lgca_state_dict'])
model.to(device)
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    """提取图像特征"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)  # transform已经返回一个PyTorch张量
        image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度，将3D转换为4D
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            _, features = model(image_tensor)
        return features.cpu().numpy()
    except Exception as e:
        print(f"特征提取错误: {str(e)}")
        raise

def compute_similarity(feat1, feat2):
    """计算特征相似度"""
    return F.cosine_similarity(torch.from_numpy(feat1), torch.from_numpy(feat2)).item()

def generate_heatmap(image_path):
    """生成注意力热力图"""
    try:
        visualizer = AttentionVisualizer(model, device)
        image = Image.open(image_path).convert('RGB')
        attention_map, _ = visualizer.generate_attention_map(image)
        
        # 保存热力图
        heatmap_filename = os.path.basename(image_path).rsplit('.', 1)[0] + '_heatmap.jpg'
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
        visualizer.visualize_attention(image, attention_map, heatmap_path)
        return heatmap_filename
    except Exception as e:
        print(f"热力图生成错误: {str(e)}")
        raise

def visualize_features(features):
    """将特征向量可视化为2D图像"""
    # 创建图像
    plt.figure(figsize=(8, 8))
    
    # 如果是单个样本，直接显示特征向量的前两个维度
    if len(features.shape) == 2 and features.shape[0] == 1:
        # 将特征向量重塑为一维数组
        features_flat = features.reshape(-1)
        # 创建柱状图显示特征分布
        plt.bar(range(min(20, len(features_flat))), features_flat[:20])
        plt.title('特征向量前20维度分布')
        plt.xlabel('维度')
        plt.ylabel('特征值')
    else:
        # 使用t-SNE降维（多个样本的情况）
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0] - 1))
        features_2d = tsne.fit_transform(features)
        plt.scatter(features_2d[:, 0], features_2d[:, 1])
        plt.title('特征向量t-SNE可视化')
    
    # 将图像转换为base64字符串
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'query_image' not in request.files:
        return jsonify({'error': '没有选择文件'})
    
    file = request.files['query_image']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    if file and allowed_file(file.filename):
        # 确保上传文件夹存在
        ensure_dir(app.config['UPLOAD_FOLDER'])
        
        # 生成安全的文件名
        filename = str(file.filename)  # 确保文件名是字符串
        query_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(query_filename)
        
        # 提取查询图片特征
        query_features = extract_features(query_filename)
        
        # 获取gallery中的所有图片
        gallery_images = []
        for gallery_filename in os.listdir(app.config['GALLERY_FOLDER']):
            if allowed_file(gallery_filename):
                gallery_path = os.path.join(app.config['GALLERY_FOLDER'], str(gallery_filename))
                gallery_features = extract_features(gallery_path)
                similarity = compute_similarity(query_features, gallery_features)
                # 只添加相似度高于75%的图片
                if similarity >= 0.75:
                    gallery_images.append({
                        'filename': gallery_filename,
                        'path': url_for('static', filename=f'gallery/{gallery_filename}'),
                        'similarity': float(similarity)
                    })
        
        # 按相似度排序
        gallery_images.sort(key=lambda x: x['similarity'], reverse=True)
        
        result = {
            'success': True,
            'query_image': url_for('static', filename=f'uploads/{filename}'),
            'gallery_images': gallery_images
        }
        
        return jsonify(result)
    
    return jsonify({'error': '不支持的文件类型'})

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': '没有选择文件'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    if file and allowed_file(file.filename):
        try:
            # 生成安全的文件名
            filename = str(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 生成热力图
            heatmap_filename = generate_heatmap(file_path)
            
            # 提取特征并可视化
            features = extract_features(file_path)
            feature_viz = visualize_features(features)
            
            result = {
                'success': True,
                'original_image': url_for('static', filename=f'uploads/{filename}'),
                'heatmap': url_for('static', filename=f'uploads/{heatmap_filename}'),
                'feature_visualization': feature_viz
            }
            
            return jsonify(result)
        except Exception as e:
            error_msg = f'处理图像时出错: {str(e)}'
            print(error_msg)
            return jsonify({'error': error_msg})
    
    return jsonify({'error': '不支持的文件类型'})

@app.route('/batch_query', methods=['POST'])
def batch_query():
    if 'query_images' not in request.files:
        return jsonify({'error': '没有选择文件'})
    
    files = request.files.getlist('query_images')
    results = []
    
    # 确保上传文件夹存在
    ensure_dir(app.config['UPLOAD_FOLDER'])
    
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue
        
        try:
            # 生成安全的文件名
            filename = str(file.filename)
            query_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(query_filename)
            
            # 提取查询图片特征
            query_features = extract_features(query_filename)
            
            # 获取gallery中的匹配图片
            matches = []
            for gallery_file in os.listdir(app.config['GALLERY_FOLDER']):
                if allowed_file(gallery_file):
                    gallery_path = os.path.join(app.config['GALLERY_FOLDER'], str(gallery_file))
                    gallery_features = extract_features(gallery_path)
                    similarity = compute_similarity(query_features, gallery_features)
                    if similarity >= 0.75:
                        matches.append({
                            'filename': gallery_file,
                            'path': url_for('static', filename=f'gallery/{gallery_file}'),
                            'similarity': float(similarity)
                        })
            
            # 按相似度排序
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 生成热力图
            heatmap_filename = generate_heatmap(query_filename)
            
            results.append({
                'query_image': url_for('static', filename=f'uploads/{filename}'),
                'heatmap': url_for('static', filename=f'uploads/{heatmap_filename}'),
                'matches': matches
            })
        except Exception as e:
            print(f"处理文件 {file.filename} 时出错: {str(e)}")
            continue
    
    return jsonify({'success': True, 'results': results})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 