import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # DON'T CHANGE THIS !!!

from flask import Flask, request, jsonify, render_template, send_from_directory
import subprocess
import uuid
import json
import traceback
import glob
import time
import threading

app = Flask(__name__)

# 配置文件存储路径
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 存储任务状态
tasks = {}

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/api/run', methods=['POST'])
def run_script():
    """API端点：运行Python脚本生成CIF文件"""
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'status': 'pending',
        'message': '任务已创建，准备执行',
        'files': [],
        'error': None
    }
    
    # 启动后台线程执行任务
    thread = threading.Thread(target=execute_script, args=(task_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'task_id': task_id,
        'status': 'pending',
        'message': '任务已提交，正在处理中'
    })

def execute_script(task_id):
    """后台执行Python脚本"""
    try:
        tasks[task_id]['status'] = 'running'
        tasks[task_id]['message'] = '正在执行脚本...'
        
        # 创建任务专属输出目录
        task_output_dir = os.path.join(OUTPUT_FOLDER, task_id)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # 执行Python脚本
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generate1.py')

        cmd = [sys.executable, script_path]
        cmd.extend(['--output_dir', task_output_dir])

        # 使用subprocess执行脚本
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = process.communicate(timeout=300)  # 设置超时时间为5分钟
        
        if process.returncode != 0:
            tasks[task_id]['status'] = 'failed'
            tasks[task_id]['message'] = '脚本执行失败'
            tasks[task_id]['error'] = stderr.decode('utf-8')
            return
        
        # 查找生成的CIF文件
        cif_files = glob.glob(os.path.join(task_output_dir, '*.cif'))
        
        if not cif_files:
            tasks[task_id]['status'] = 'failed'
            tasks[task_id]['message'] = '未生成CIF文件'
            return
        
        # 更新任务状态
        file_list = []
        for cif_file in cif_files:
            file_name = os.path.basename(cif_file)
            file_list.append({
                'name': file_name,
                'path': f'/api/files/{task_id}/{file_name}'
            })
        
        tasks[task_id]['status'] = 'completed'
        tasks[task_id]['message'] = f'成功生成 {len(cif_files)} 个CIF文件'
        tasks[task_id]['files'] = file_list
        
    except Exception as e:
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['message'] = '执行过程中发生错误'
        tasks[task_id]['error'] = str(e) + '\n' + traceback.format_exc()

@app.route('/api/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """API端点：获取任务状态"""
    if task_id not in tasks:
        return jsonify({'error': '任务不存在'}), 404
    
    return jsonify(tasks[task_id])

@app.route('/api/files/<task_id>/<filename>', methods=['GET'])
def get_file(task_id, filename):
    """API端点：获取生成的文件"""
    task_output_dir = os.path.join(OUTPUT_FOLDER, task_id)
    return send_from_directory(task_output_dir, filename)

@app.route('/api/files', methods=['GET'])
def list_all_files():
    """API端点：列出所有生成的文件"""
    all_files = []
    for task_id, task_info in tasks.items():
        if task_info['status'] == 'completed':
            all_files.extend(task_info['files'])
    
    return jsonify(all_files)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
