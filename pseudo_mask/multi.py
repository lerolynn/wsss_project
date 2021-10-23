import os
from main import make_cam


def main():
    data_dir = "../data/train"
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            print(filename)
            make_cam(data_dir, os.path.join(data_dir, filename), "base_model.layer4", 3, "./results", True)


if __name__ == "__main__":
    main()
