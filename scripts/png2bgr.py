import sys

import cv2

IMAGE_OUTPUT_SIZE = (512, 1024)


def png2bgr(png_file):
    img = cv2.imread(png_file)
    print("Original image size: ", img.shape)
    img = cv2.resize(img, (IMAGE_OUTPUT_SIZE[1], IMAGE_OUTPUT_SIZE[0]))
    print("Resized image size: ", img.shape)
    ext = png_file.split('.')[-1]
    output_name = png_file.replace(ext, "bgr")

    with open(output_name, "wb") as fout:
        for c in range(3):
            for h in range(IMAGE_OUTPUT_SIZE[0]):
                for w in range(IMAGE_OUTPUT_SIZE[1]):
                    value = int(img[h][w][c]).to_bytes(length=1, byteorder="big", signed=False)
                    fout.write(value)

    return


if __name__ == '__main__':
    assert 2 == len(sys.argv), "Usage $0 img_file_path"
    png_file = sys.argv[1]
    png2bgr(png_file)
