import os

for crop in ['groundnut', 'wheat']:
    path = fr'D:\crop-disease-advisor\data\{crop}'
    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        continue
    print(f'\n{crop.upper()} classes:')
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
            print(f'  {cls}: {count} images')
