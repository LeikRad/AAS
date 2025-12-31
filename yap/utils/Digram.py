from .bin2int import bin2int
import numpy as np

class Digram:
    def __init__(self):
        self.first = None
        self.second = None
        self.texture_array = np.zeros((256, 256, 2), dtype=np.float32)
                
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
        self.texture_array[bin2int(self.first), bin2int(self.second), 0] += 1
        self.texture_array[bin2int(self.first), bin2int(self.second), 1] += self.bitCounter
            
    def finalize(self):
        # update second channel to average position
                    
        # Normalize the values to floating-point by dividing by data size
        data_size = self.bitCounter + 1 # Total number of bits processed
        # Channel 0: normalize occurrence count
        
        self.texture_array[:, :, 0] /= data_size
        self.texture_array[:, :, 1] /= (data_size*data_size) 


    def getTextureArray(self):
        return self.texture_array
