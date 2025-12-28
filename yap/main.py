from utils import FileBinaryObject, Digram, bin2int, ImageWriter
import matplotlib.pyplot as plt

def main():
    file_path = '2513228.2513294.pdf'
    binary_object = FileBinaryObject(file_path)
    finaldigram = proccess_binary_file(binary_object)

    # plot occurences to matplotlib
    plot_data = finaldigram.getPlotReadable()
    image_writer = ImageWriter()
    image_writer.set_pixel(plot_data)
    image_writer.save_image('output.png')


def proccess_binary_file(binary_object):
    binary_object.open()
    tempDigram = Digram()
    try:
        while True:
            byte = binary_object.readNBytes(1)
            if not byte:
                break
            tempDigram.addNextBit(byte)
    finally:
        binary_object.close()
    return tempDigram

if __name__ == "__main__":
    main()
