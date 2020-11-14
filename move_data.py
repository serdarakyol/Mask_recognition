import os, random
from PIL import Image

DIRECTORY = r"/home/ak/Desktop/mask_recognition/archive/data/with_mask"
data = []
while 1:
    check_data = random.choice(os.listdir(DIRECTORY))
    if check_data not in data:
        data.append(check_data)
        if len(data)==600:
            break


save_dir = "./with_mask"
for datum in data:
    img_get_path = os.path.join(DIRECTORY, datum)
    img_save_path = os.path.join(save_dir, datum)
    img = Image.open(img_get_path)
    img = img.convert('RGB')
    img = img.save(img_save_path)
    print(f"Moved file==> {datum}")