from PIL import Image

class ImageWriter:
    def __init__(self, width=256, height=256):
        self.width = width
        self.height = height
        self.pixels = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]

    def set_pixel(self, plotData):
        for i in range(len(plotData[0])):
            x = plotData[1][i]
            y = plotData[0][i]
            if 0 <= x < self.width and 0 <= y < self.height:
                self.pixels[x][y] = plotData[2][i]
                
    def save_image(self, filename = 'output.png'):
        img = Image.new('RGB', (self.width, self.height))
        for y in range(self.height):
            for x in range(self.width):
                img.putpixel((x, self.height-1-y), self.pixels[x][y])
        img.save(filename)
        print(f"Image saved as {filename}")
    
    