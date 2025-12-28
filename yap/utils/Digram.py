from .bin2int import bin2int

class Digram:
    def __init__(self):
        self.first = None
        self.second = None
        self.occurrences = {}
        self.occurrences_index = {}
        self.bitCounter = 0

    def addNextBit(self, bit):
        if self.first is None:
            self.first = bit
            return
        elif self.second is None:
            self.second = bit
        else:
            self.first = self.second
            self.second = bit
        self.addOccurrence()
        self.bitCounter += 1
    
    def addOccurrence(self):
        key = (self.first, self.second)
        if key in self.occurrences:
            self.occurrences[key] += 1
            self.occurrences_index[key].append(self.bitCounter)
        else:
            self.occurrences[key] = 1
            self.occurrences_index[key] = [self.bitCounter]

    def getOccurrences(self):
        return self.occurrences

    def getOccurrencesIndex(self):
        return self.occurrences_index
    
    def getPlotReadable(self):
        plot_data_bin = self.getOccurrences().keys()
        plot_data_readable = []
        rgb_colors = []
        for key in plot_data_bin:
            key_readable = (bin2int(n) for n in key)
            plot_data_readable.append(tuple(key_readable))
            rgb_colors.append(self.getColor(self.occurrences_index[key]))
        # transform [(x1, y1), (x2, y2), ...] to [(x1, x2, ...), (y1, y2, ...)] 
        plot_data_transformed = list(zip(*plot_data_readable))
        plot_data_transformed.append(rgb_colors)
        
        return plot_data_transformed
    
    def getColor(self, indexes):
        def lerp(a, b, t):
            return a + (b - a) * t
        
        orange = (255, 165, 0)
        blue = (0, 0, 255)
        white = (50, 50, 50)
        
        avg_appearance = sum(indexes) / len(indexes)
        t = avg_appearance / self.bitCounter
        t = max(0.0, min(1.0, t))  # Clamp t to [0, 1]
        
        if t < 0.5:
            u = t * 2
            r = int(lerp(orange[0], white[0], u))
            g = int(lerp(orange[1], white[1], u))
            b = int(lerp(orange[2], white[2], u))
        else :
            u = (t - 0.5) * 2
            r = int(lerp(white[0], blue[0], u))
            g = int(lerp(white[1], blue[1], u))
            b = int(lerp(white[2], blue[2], u))
        
        return (r, g, b)