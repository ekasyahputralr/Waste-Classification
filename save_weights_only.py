import os
from kamis import train_model  # atau dari file/module tempat model dilatih

def save_final_weights():
    model = train_model()  # pastikan ini tidak memicu training ulang!
    
    model_path = 'D:\\testmatterport\\Mask-RCNN-TF2\\logs\\waste_final.h5'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model.keras_model.save_weights(model_path)
    print(f"Model weights berhasil disimpan di: {model_path}")

if __name__ == "__main__":
    save_final_weights()
