# YOLOv8 画像認識デモアプリ

このプロジェクトは、Ultralytics の YOLOv8 モデルを用いた画像認識のデモアプリです。 Tkinter による簡易 GUI 上で、指定フォルダ内の画像を順番に読み込み、物体検出結果を表示します。

🔧 使用技術

Python 3.x

Ultralytics YOLOv8

OpenCV (cv2)

Pillow (PIL)

Tkinter（GUI）

ディレクトリ構成
project_root/
├── yolo_image_detector.py   # 本体スクリプト
├── yolo/                    # 判定対象の画像フォルダ
│   ├── test1.jpg
│   └── test2.png

実行方法
pip install ultralytics opencv-python pillow
python yolo_image_detector.py

🖼️ アプリの動作

アプリを起動すると、GUIウィンドウが表示されます。

「判定」ボタンを押すと、画像を1枚ずつ読み込み表示。

YOLOv8 による物体検出を行い、OpenCVウィンドウに結果を表示。

コンソールにも検出物体とその位置情報を出力。

🔍 出力例（コンソール）
--- 検出結果 ---
検出ボックス数: 2
物体名: cat
位置: X=140.5, Y=102.3, 幅=83.2, 高さ=75.1
物体名: dog
位置: X=311.7, Y=219.8, 幅=115.6, 高さ=92.3
--------------

 補足

使用モデルは yolov8x.pt（Ultralytics 提供の事前学習済モデル）

conf=0.5 の設定により小さな物体検出はスキップされます

結果表示ウィンドウは cv2.imshow() による一時表示です

ライセンス・利用

このコードは学習・デモ・ポートフォリオ用途に自由に使用可能です。

商用利用の場合は Ultralytics のライセンスに準拠してください。
