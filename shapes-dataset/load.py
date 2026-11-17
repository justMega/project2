import os


name2label = {
    "circle": 0,
    "square": 1,
    "fig_eight": 2,
    "rectangle": 3,
    "triangle": 4,
    "elipse": 5
}

paths = []
labels = []

root = "2D_Geometric_Shapes_Dataset"
for folder in os.listdir(root):
    for file in os.listdir(f"{root}/{folder}"):
        paths.append(f"{root}/{folder}/{file}")
        labels.append(name2label[folder])

print(paths[:30])
print(labels[:30])
