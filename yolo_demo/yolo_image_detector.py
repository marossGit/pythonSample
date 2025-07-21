# yolo_image_detector.py
# -------------------------
# 本コードは Ultralytics YOLOv8 モデルを用いた簡易画像判定アプリのサンプルです。
# GUI上で画像を順に表示しながら、YOLOモデルで物体検出を行い、結果を表示します。

import tkinter
import os
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO  # YOLOv8ライブラリのインポート

filePath = "./yolo/"  # 画像ファイル格納ディレクトリ
size = 300  # 表示ウィンドウサイズ

class JudgeApp:
    def __init__(self):
        # GUIウィンドウの作成
        window = tkinter.Tk()
        window.title("YOLO画像認識デモ")

        # Canvas領域（画像表示用）
        self.canvas = tkinter.Canvas(window, bg="black", width=size, height=size)
        self.canvas.pack()

        # ボタンの作成
        detect_button = tkinter.Button(window, text="判定", command=self.on_judge)
        detect_button.pack(side=tkinter.RIGHT)

        # ファイルリスト読み込みと初期化
        self.files = [f for f in os.listdir(filePath) if f.endswith(('.jpg', '.png'))]
        self.counter = 0

        # YOLOv8モデルの読み込み（事前学習済モデル yolov8x）
        self.model = YOLO('yolov8x.pt')

        # GUIループ開始
        window.mainloop()

    def on_judge(self):
        """
        判定ボタンが押されたときに呼ばれる関数。
        画像を読み込み、YOLOで物体検出を行い、結果をGUIとOpenCVで表示。
        """
        # ファイル名取得
        file_name = filePath + self.files[self.counter]
        self.counter = (self.counter + 1) % len(self.files)  # 次の画像へ

        try:
            image = Image.open(file_name)
        except:
            print(f"読み込み失敗: {file_name}")
            return

        # 画像をtkinter表示用にリサイズ＆表示
        self.tk_image = ImageTk.PhotoImage(image.resize((250, 250)))
        self.canvas.create_image(0, 50, anchor="nw", image=self.tk_image)

        # YOLOv8で予測（conf=0.5で小さな物体は除外）
        result = self.model.predict(image.resize((500, 500)), conf=0.5, show=False)[0]
        img = result.plot()  # 検出結果を画像として描画

        # OpenCVで縮小表示（例：70%）
        resized_img = cv2.resize(img, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow("YOLO Result", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 結果をコンソールにも出力
        print("--- 検出結果 ---")
        print(f"検出ボックス数: {len(result.boxes)}")
        for box in result.boxes:
            pos = box.xywh[0]
            class_id = box.cls[0].item()
            class_name = result.names[class_id]
            print(f"物体名: {class_name}")
            print(f"位置: X={pos[0]:.1f}, Y={pos[1]:.1f}, 幅={pos[2]:.1f}, 高さ={pos[3]:.1f}")
        print("--------------\n")

# アプリ実行
if __name__ == "__main__":
    JudgeApp()
