# 视觉计算与深度学习期末课程项目
- 张听扬 2100012962
- 陈梓航 2100012948

## 环境
- 在Windows环境下运行
- 虚拟环境要求python版本>=3.8,如果未安装taichi,可使用以下代码安装taichi库:
```sh
pip install taichi
```

## 运行
- 运行文件run.ipynb
- 依次运行所有单元格，每个单元格的主要作用在首行的备注中
- 每完成一轮训练，都会在渲染器中更新一次图像，即可以以固定视角预览最新模型下的渲染效果，需要注意，在训练结束前，这一窗口是不可交互的。
- 训练结束后，可以在窗口中拖动鼠标来调整视角，按下鼠标右键会保存当前的图像。
- 改变初始化方式：在initialize voxel system and buffer部分，可以切换vs的初始化方法（可以将zero_init改为random_init，这将使得grid随机初始化）
- 切换数据集：在load data部分改变load函数中参数可以改变数据集（我们提供了'Lego','Hotdog','Drums','Chair'供训练与对比）
-  结果对比：在val()函数中，会随机选择一张测试集中的图片，并将原图和项目产生的图输出到out文件夹中，同时输出渲染图与原图相比的psnr与ssim值
-  查看状态：在vs初始化后，任意时刻调用以下代码可以查看当前体素状态：
```python
camera = Camera(vs)
while camera.gui.running:
    camera.cal_rays()
    camera.render()
    camera.display()
    camera.processEvents()
```
- 调用以下代码可以查看上一次渲染的结果：
```python
gui = ti.GUI('window', (800,800), fast_gui=False)
while gui.running:
    gui.set_image(buffer)
    gui.show()
```