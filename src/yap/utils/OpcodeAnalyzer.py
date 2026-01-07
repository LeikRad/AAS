import lief
from capstone import *
import hashlib
import numpy as np
from PIL import Image

class OpcodeAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.marked_opcodes = ("mov", "ret")
        self.opcode_string = ""
        self.n = 10
        self.texture_array = np.zeros((2**self.n, 2**self.n, 3), dtype=np.float32)

    def analyze(self):
        binary = lief.parse(self.filepath)

        md = Cs(CS_ARCH_X86, CS_MODE_64)
        md.detail = True

        for section in binary.sections:            
            code = bytes(section.content)
            addr = section.virtual_address
            
            for insn in md.disasm(code, addr):
                word = insn.mnemonic
                if len(word) > 3:
                    word = word[:3]
                self.opcode_string += word
                
                if insn.mnemonic in self.marked_opcodes:
                    self.proccess_string()
                    self.opcode_string = ""
                        
    def proccess_string(self):
        simHash = self.simHash_string()       
        dfb2Hash = self.djb2_hash()
        
        x, y = self.hash_to_coords(simHash)
        r, g, b = self.hash_to_color(dfb2Hash)
        
        # splash in neighbor pixels
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < (1 << self.n) and 0 <= ny < (1 << self.n):
                    self.texture_array[nx, ny, 0] = min(self.texture_array[nx, ny, 0] + r / 255.0, 1.0)
                    self.texture_array[nx, ny, 1] = min(self.texture_array[nx, ny, 1] + g / 255.0, 1.0)
                    self.texture_array[nx, ny, 2] = min(self.texture_array[nx, ny, 2] + b / 255.0, 1.0)
                
    def clamp_coordinates(self, x, y):
        """Clamp coordinates to image bounds instead of wrapping"""
        size = (1 << self.n) - 1  # 2^n - 1
        x = max(0, min(x, size))
        y = max(0, min(y, size))
        return (x, y)

    def hash_to_coords(self, simHash):
        mask = (1 << self.n) - 1
        x = simHash & mask
        y = (simHash >> self.n) & mask
        return self.clamp_coordinates(x, y)
        
    def hash_to_color(self, djb2Hash):
        r = djb2Hash  & 0xFF
        g = (djb2Hash >> 8) & 0xFF
        b = (djb2Hash >> 16) & 0xFF
        return r, g, b
    
    def simHash_string(self):
        """Compute SimHash for the opcode string"""
        if not self.opcode_string:
            return
        
        # SimHash implementation
        hash_bits = 64
        v = [0] * hash_bits
        
        
        # Hash each token using MD5
        h = int(hashlib.md5(self.opcode_string.encode()).hexdigest(), 16)
        
        # Update bit vector
        for i in range(hash_bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
        
        # Generate final hash
        simhash = 0
        for i in range(hash_bits):
            if v[i] > 0:
                simhash |= (1 << i)
        return simhash
    
    def djb2_hash(self):
        """Compute DJB2 hash for the opcode string"""
        hash = 5381
        for c in self.opcode_string:
            hash = ((hash << 5) + hash) + ord(c)  # hash * 33 + c
        return hash & 0xFFFFFFFFFFFFFFFF  # Return as 64-bit hash
    
    def save_image(self, filepath):
        """Save texture array as image"""
        # Convert float32 (0.0-1.0) to uint8 (0-255)
                    
        img_array = (self.texture_array * 255).astype(np.uint8)
        
        # Create PIL image and save it
        img = Image.fromarray(img_array, 'RGB')
        img.save(filepath)
        print(f"Image saved to {filepath}")
    
