# 第一屆深度學習(Deep Learning)與電腦視覺(Computer Vision)馬拉松
## 1st-DL-CVMarathon

### 基礎影像處理
學習影像處理基礎，並熟悉 OpenCV 寫作方式以及如何前處理<br><br>

1.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day001_read_image_HW.ipynb'>OpenCV 簡介 + 顯示圖片</a>: 入門電腦視覺領域的重要套件: OpenCV<br>
2.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day002_change_color_space_HW.ipynb'>Color presentation 介紹 (RGB, LAB, HSV)</a>:  淺談圖片不同的表示方式<br>
3.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day003_color_spave_op_HW.ipynb'>顏色相關的預處理 (改變亮度, 色差)</a>:  透過圖片不同的表示方式實作修圖效果<br>
4.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day004_geometric_transform_HW.ipynb'>以圖片為例做矩陣操作 (翻轉, 縮放, 平移)</a>:  淺談基礎的幾合變換: 翻轉 / 縮放 / 平移<br>
5.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day005_draw_HW.ipynb'>透過 OpenCV 做圖並顯示 (長方形, 圓形, 直線, 填色)</a>:  實作 OpenCV 的小畫家功能<br>
6.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day006_affine_HW.ipynb'>affine transformation 概念與實作</a>:  仿射轉換的入門與實作: affine transform<br>
7.perspective transformation 概念與實作:  視角轉換的入門與實作: perspective transform<br>
8.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day008_sobel_gaussian_blur_HW.ipynb'>Filter 操作 (Sobel edge detect, Gaussian Blur)</a>:  初探邊緣檢測與模糊圖片操作: 了解 filter 的運用<br>
9.SIFT 介紹與實作 (feature extractor):  SIFT: 介紹與實作經典的傳統特徵<br>
10.SIFT 其他應用 (keypoint matching):  SIFT 案例分享: 特徵配對<br>
<br><br>

### 電腦視覺深度學習基礎
打好卷積神經網路的概念，並了解 CNN 各種代表性的經典模型<br><br>

11.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day011_CNN-%E8%A8%88%E7%AE%97%E5%8F%83%E6%95%B8%E9%87%8F_HW.ipynb'>CNN分類器架構：卷積層</a>:  卷積是CNN的核心，了解卷積如何運行 就能幫助我們理解CNN的原理<br>
12.CNN分類器架構：步長、填充:  填充與步長是CNN中常見的超參數， 了解如何設置能幫助我們架構一個CNN Model<br>
13.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day013_%E6%B1%A0%E5%8C%96%E3%80%81%E5%85%A8%E9%80%A3%E6%8E%A5%E5%B1%A4_HW.ipynb'>CNN分類器架構：池化層、全連接層</a>:  池化層時常出現於CNN結構中，而FC層則會接在模型輸出端， 了解如兩者用途能幫助我們架構一個CNN Model<br>
14.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day014_Batch%2BNormalization_HW.ipynb'>CNN分類器架構：Batch Normalization</a>:  Batch Normalization出現在各種架構中， 了解BN層能解決怎樣的問題是我們本章的重點<br>
15.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day015_Cifar_HW.ipynb'>訓練一個CNN分類器：Cifar10為例</a>:  綜合上述CNN基本觀念， 我們如何結合這些觀念打造一個CNN 模型<br>
16.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day016_Image%2BAugmentation_HW.ipynb'>如何使用Data Augmentation</a>:  訓練模型時常常會遇到資料不足的時候，適當的使用Image Augmentation能提升模型的泛化性<br>
17.AlexNet:  綜合之前所學的CNN觀念，認識第一個引領影像研究方向朝向深度學習的模型<br>
18.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day018_Vgg16_HW.ipynb'>VGG16 and 19</a>:  模型繼續進化，認識簡單卻又不差的CNN模型<br>
19.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day019_Inception_HW.ipynb'>InceptionV1-V3</a>:  Inception module提供大家不同於以往的思考方式，將模型的參數量減少，效能卻提升了許多<br>
20.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day020_Classic%2BCNN-ResNet%E3%80%81InceptionV4%E3%80%81Inception-ResNet_HW.ipynb'>ResNetV1-V2、InceptionV4、Inception-ResNet</a>:  首次超越人類分類正確率的模型，Residual module也影響了後來許多的模型架構<br>
21.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day021_Transfer%2BLearning_HW.ipynb'>Transfer learning</a>:  學習如何利用前人的知識輔助自己訓練與跨領域學習的方法<br>
22.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day022_Captcha_HW.ipynb'>Breaking Captchas with a CNN</a>:  了解如何使用CNN+CTC判定不定長度字串<br>
<br><br>

### CNN 應用案例學習
學習目前最常使用的 CNN 應用案例：YOLO 物件偵測實務完全上手<br><br>

#### Object detection
23.Object detection原理:  了解Object Detection出現的目的與基本設計原理<br>
24.Object detection基本介紹、演進:  了解Object Detection一路發展下來，是如何演進與進步<br>
25.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day025_IOU_HW.ipynb'>Region Proposal、IOU概念</a>:  IOU是貫穿Object Detection的一個重要觀念，了解如何計算IOU對了解Object Detection中許多重要步驟會很有幫助<br>
26.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day026_RPN_HW.ipynb'>RPN架構介紹</a>:  RPN是Faster RCNN成功加速的關鍵，了解RPN便能深入認識Faster RCNN<br>
27.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day027_BBOX%2BRegression_HW.ipynb'>Bounding Box Regression原理</a>:  所有的Object Detection模型都需要做Bounding Box的Regression，了解其是如何運作的能幫助我們更認識Object Detection<br>
28.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day028_NMS_HW.ipynb'>Non-Maximum Suppression (NMS)原理</a>:  所有的Object Detection模型都有Non Maximum Suppression的操作，了解其是如何運作的能幫助我們更認識Object Detection<br>
29.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/tree/master/homework/Day029-031_Object%20Detection%20%E7%A8%8B%E5%BC%8F%E5%B0%8E%E8%AE%80_HW'>程式導讀、實作</a>:  了解如何搭建一個SSD模型<br>
<br>

#### YOLO
32.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day32_yolo_prediction_HW.ipynb'>YOLO 簡介及算法理解</a>:  了解 YOLO 的基本原理<br>
33.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day33_YOLO%2B%E7%B4%B0%E7%AF%80%E7%90%86%E8%A7%A3%2B%E7%B6%B2%E8%B7%AF%E8%BC%B8%E5%87%BA%E7%9A%84%E5%BE%8C%E8%99%95%E7%90%86_HW.ipynb'>YOLO 細節理解 - 網路輸出的後處理</a>:  理解網路輸出的後處理，執行nms的過程<br>
34.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day34_YOLO%2B%E7%B4%B0%E7%AF%80%E7%90%86%E8%A7%A3%2B%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8_HW.ipynb'>YOLO 細節理解 - 損失函數</a>:  認識YOLO損失函數設計架構與定義<br>
35.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day35_yolo_loss.ipynb'>YOLO 細節理解 - 損失函數程式碼解讀</a>:  講解YOLO損失函數程式碼<br>
36.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day36_YOLO_%E7%B4%B0%E7%AF%80%E7%90%86%E8%A7%A3-%E7%B6%B2%E7%B5%A1%E6%9E%B6%E6%A7%8B_HW.ipynb'>YOLO 細節理解 - 網路架構</a>:  了解YOLO網絡架構的設計與原理<br>
37.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day37_%E7%B6%B2%E7%B5%A1%E6%9E%B6%E6%A7%8B%E7%A8%8B%E5%BC%8F%E7%A2%BC_HW.ipynb'>YOLO 細節理解 - 網路架構程式碼解讀</a>:  講解YOLO網絡架構程式碼<br>
38.YOLO 演進:  簡單了解 YOLO 的演進<br>
39.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day39_yolov3_keras_HW.ipynb'>使用 YOLOv3 偵測圖片及影片中的物件</a>:  了解如何基於現有的 YOLOv3 程式碼客制化自己的需求<br>
40.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day40_tiny_yolov3_keras_HW.ipynb'>更快的檢測模型 - tiny YOLOv3</a>:  了解如何使用 tiny YOLOv3 來滿足對檢測速度的需求<br>
41.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day41_train_yolov3_HW.ipynb'>訓練 YOLOv3</a>:  了解如何訓練 YOLOv3 檢測模型<br>
<br><br>

### 電腦視覺深度學習實戰
人臉關鍵點檢測及其應用<br><br>

42.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day42_explore_facial_keypoint_data_HW.ipynb'>人臉關鍵點-資料結構簡介</a>:  探索 kaggle 臉部關鍵點資料<br>
43.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day43_define_network_HW.ipynb'>人臉關鍵點-檢測網路架構</a>:  學習用 keras 定義人臉關鍵點檢測的網路架構<br>
44.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day44_train_facial_keypoint_HW.ipynb'>訓練人臉關鍵點檢測網路</a>:  體會訓練人臉關鍵點檢測網路的過程<br>
45.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day45_facial_keypoint_application.ipynb'>人臉關鍵點應用</a>:  體驗人臉關鍵點的應用 - 人臉濾鏡<br>
46.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day046_MobileNet_HW.ipynb'>Mobilenet</a>:  輕量化模型簡介 (MobileNet)<br>
47.<a href='https://github.com/PrestonYU/1st-DL-CVMarathon/blob/master/homework/Day047_MobileNetv2_HW.ipynb'>Ｍobilenetv2</a>:  MobileNet v2 簡介<br>
48.Tensorflow Object Detection API:  Tensorflow Object Detection API使用方式<br>
<br>
