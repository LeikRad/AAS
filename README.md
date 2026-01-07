# Visual Malware Detection via Convolutional Neural Networks on Digram and Opcode Representations

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)

## Introduction

This repository contains the source code for the Practical Project of the **Aprendizagem Aplicada à Segurança (AAS)** course.

The objective of this project is to validate the effectiveness of using visual representations of binary files for passive malware detection versus traditional methods like similarity analysis. We implement two visualization techniques: digram analysis and opcode analysis, and train Convolutional Neural Networks (CNNs) to classify the generated images as benign or malicious.

**Authors:**
* Bernardo Figueiredo (108073)
* Leonardo Falcão (127891)

## Prerequisities

* Python 3.8 or higher
* Pip (Python Package Installer)

## Installation

It is strongly recommended to run this project inside a virtual environment to avoid dependency conflicts.

### 1. Extract the compressed file
```bash
tar xvf [compressed_file].tar.xz 
cd [your-project]
```

### For each folder in the src folder!
### 2. Create and Activate Virtual Environment

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Dataset

This project utilizes a handmade dataset consiting of image files created by our visualization pipeline, which processes binary files into visual representations using digram and opcode analysis techniques.

  * **Source:** Handmade dataset.

## Usage

To create visualizations from files:

First place the files into your desired folder structure inside the `/yap/input/` directory. (if the input folder has more than one file, it will process all files inside it)

Then run the following command:
```bash
python src/yap/main.py
```

To train the model, place the generated images into the `/CNN/dataset/` directory, following the structure:
```
CNN/data_dir
├── anomaly
│   ├── file1.png
│   ├── file2.png
│   └── ...
└── normal
    ├── file1.png
    ├── file2.png
    └── ...
```

Then run the training script:
```bash
python src/CNN/binary_anomaly_detector.py 
```

To run the classification on a new file:

```bash
python src/CNN/test.py -i path/to/file.exe 
```

## Major Results

No results showed an statistically significant improvement over random guessing due to the small dataset size. However, some promising patterns were observed.

**Key Findings:**

  * Statistical analysis of digram/opcode visualizations were better than our CNN models, although with a higher dataset more promising results could be achieved.
  * The CNN models were able to learn some meaningful patterns, but overfitting was a significant issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.