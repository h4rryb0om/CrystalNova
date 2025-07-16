# 晶体结构生成与可视化网站部署指南

## 项目概述

这是一个基于Flask的Web应用，允许用户通过点击按钮运行晶体结构生成代码，并在网页中可视化生成的.cif文件。该应用使用3Dmol.js库提供交互式3D可视化功能。

## 项目结构

```
crystal_generator/
├── requirements.txt      # 项目依赖
├── src/
│   ├── main.py           # Flask应用主入口
│   ├── generate1.py      # 用户提供的原始Python代码
│   ├── generate_cif.py   # 用于生成CIF文件的脚本
│   ├── static/           # 静态资源目录
│   │   ├── uploads/      # 上传文件存储目录
│   │   └── outputs/      # 生成的CIF文件存储目录
│   └── templates/        # HTML模板目录
│       └── index.html    # 主页面模板
```

## 依赖项

项目依赖项已在`requirements.txt`文件中列出，主要包括：

- Flask: Web框架
- PyMatGen: 材料基因组学Python库
- PyTorch: 深度学习框架
- NumPy: 科学计算库
- Torch-Geometric: 图神经网络库

## 部署步骤

### 1. 安装依赖

```bash
# 创建并激活虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行应用

```bash
# 进入src目录
cd crystal_generator/src

# 运行Flask应用
python main.py
```

应用将在`http://localhost:5000`启动，可以通过浏览器访问。

## 使用说明

1. 打开浏览器访问`http://localhost:5000`
2. 点击"生成晶体结构"按钮开始生成CIF文件
3. 等待生成完成，生成的文件将显示在左侧列表中
4. 点击文件名可在右侧查看器中加载并可视化该结构
5. 使用"球棍模型"、"空间填充模型"和"线框模型"按钮切换不同的显示模式
6. 点击"下载当前文件"可下载当前查看的CIF文件

## 注意事项

- 生成过程可能需要一些时间，取决于服务器性能
- 如果遇到依赖安装问题，特别是PyTorch和Torch-Geometric，请参考它们的官方安装指南
- 确保服务器有足够的内存运行深度学习模型

## 生产环境部署

对于生产环境，建议：

1. 使用Gunicorn或uWSGI作为WSGI服务器
2. 配置Nginx作为反向代理
3. 设置适当的安全措施，如HTTPS

示例Gunicorn启动命令：

```bash
gunicorn -w 4 -b 0.0.0.0:5000 main:app
```

## 故障排除

- 如果遇到"模块未找到"错误，请检查依赖是否正确安装
- 如果生成过程失败，请检查错误详情，可能是内存不足或GPU相关问题
- 如果3D可视化不显示，请确保浏览器支持WebGL
