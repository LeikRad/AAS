from utils import FileBinaryObject, Digram, OpcodeAnalyzer, ShaderRenderer
import matplotlib.pyplot as plt
import os

def main():
    # load input directory
    root_input_dir = 'input/'
    
    # recursively discover files in input directory
    files_path = []
    
    for root, dirs, files in os.walk(root_input_dir):
        for file in files:
            if file.startswith('.'):
                continue
            files_path.append(os.path.join(root, file))

        # mimick folder structure in output directory
        for dir in dirs:
            output_dir = os.path.join('output/', os.path.relpath(os.path.join(root, dir), root_input_dir))
            os.makedirs(output_dir, exist_ok=True)
     
    execute_files = ["exe", "dll", "elf", "bin", "elf64", "elf32"]
    
    for file_path in files_path:
        print(f'Processing file: {file_path}')
        binary_object = FileBinaryObject(file_path)
        finaldigram = proccess_binary_file(binary_object)
        
        # Save as image using ImageWriter
        
        output_image_path = os.path.join('output/', os.path.relpath(file_path, root_input_dir))
        output_image_path = os.path.splitext(output_image_path)[0] + '.png'
        
        # if executable, do opcode analysis too
        if any(file_path.lower().endswith(ext) for ext in execute_files):
            # create new folder inside output for opcode images
            opcode_path = os.path.join(os.path.dirname(output_image_path), 'opcode_images')
            os.makedirs(opcode_path, exist_ok=True)
            opcode_image_path = os.path.join(opcode_path, os.path.basename(output_image_path))
            
            opcode_analyzer = OpcodeAnalyzer(file_path)
            opcode_analyzer.analyze()
            opcode_analyzer.save_image(opcode_image_path)
            
            # create new folder inside output for digram images
            output_image_path = os.path.join(os.path.dirname(output_image_path), 'digram_images')
            os.makedirs(output_image_path, exist_ok=True)
            output_image_path = os.path.join(output_image_path, os.path.basename(opcode_image_path))
            
        shader_renderer = ShaderRenderer(width=256, height=256)
    
        # Upload the texture array to GPU
        shader_renderer.upload_texture(finaldigram.texture_array)

        # Render and save the image
        shader_renderer.save_image(output_image_path)

        # Cleanup
        shader_renderer.cleanup()


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
        tempDigram.finalize()
        binary_object.close()
    return tempDigram

if __name__ == "__main__":
    main()
