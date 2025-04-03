import json



annotations_file = "D:/testmatterport/Mask-RCNN-TF2/datasetbaru/train/_annotations.coco.json"

with open(annotations_file, "r") as f:
    data = json.load(f)

print(type(data))  # Apakah list atau dictionary?
print(data.keys() if isinstance(data, dict) else data[:2])  # Cetak sebagian data untuk cek struktur
