# 基础镜像
FROM python:3.9-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 先复制依赖文件（利用Docker缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src

# 设置环境变量
ENV NUM_STRUCTURES=3 \
    BATCH_SIZE=1 \
    PORT=5000

# 暴露端口
EXPOSE $PORT

# 启动命令
CMD ["gunicorn", "src.main:app", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--timeout", "300"]