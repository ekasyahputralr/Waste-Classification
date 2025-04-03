# Contoh Bounding Box yang diberikan
bbox = {
    "id": "1",
    "label": "anorganik",
    "x": "303.74",  # Konversikan ke float
    "y": "359.53",  # Konversikan ke float
    "width": "267.47",  # Konversikan ke float
    "height": "255.39"  # Konversikan ke float
}

# Konversikan ke tipe data float
x = float(bbox["x"])
y = float(bbox["y"])
width = float(bbox["width"])
height = float(bbox["height"])

# Bounding box siap digunakan
print(x, y, width, height)

# Contoh titik polygon mask
polygon_points = [
    [188, 251],
    [194, 343],
    [193, 388],
    [205, 394],
    [211, 406],
    [362, 477],
    [385, 426],
    [398, 406],
    [424, 305],
    [289, 263],
    [212, 248],
    [190, 247]
]

# Pastikan titik-titik berada dalam format tuple (int, int)
polygon_points = [(int(x), int(y)) for x, y in polygon_points]
print(polygon_points)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar
image = cv2.imread("D:/testmatterport/Mask-RCNN-TF2/datasetbaru/train/bungkus_makanan-43-_jpg.rf.4b0a5c56125b4a656b5de728e7086c6c.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Mengonversi ke format RGB

# Menggambar Bounding Box
start_point = (int(x), int(y))  # Titik kiri atas dari bounding box
end_point = (int(x + width), int(y + height))  # Titik kanan bawah dari bounding box
color = (0, 255, 0)  # Warna bounding box (hijau)
thickness = 2  # Ketebalan garis bounding box
image = cv2.rectangle(image, start_point, end_point, color, thickness)

# Menggambar Polygon Mask
mask_points = np.array(polygon_points, np.int32)  # Konversi titik polygon ke format numpy
mask_points = mask_points.reshape((-1, 1, 2))  # Format untuk OpenCV
image = cv2.fillPoly(image, [mask_points], (255, 0, 0))  # Mengisi polygon dengan warna biru

# Menampilkan gambar menggunakan matplotlib
plt.imshow(image)
plt.axis('off')  # Menyembunyikan axis
plt.show()
