# TopoOOD: Graph Out-of-Distribution Detection Goes Neighborhood Shaping 🌐

![TopoOOD](https://img.shields.io/badge/TopoOOD-Graph_Out_of_Distribution_Detection-brightgreen)

Welcome to the official implementation of the ICML24 paper titled **"Graph Out-of-Distribution Detection Goes Neighborhood Shaping."** This repository serves as a resource for researchers and practitioners interested in the intersection of graph theory and out-of-distribution detection.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)
8. [Contact](#contact)

## Introduction

Out-of-distribution (OOD) detection is a crucial task in machine learning, particularly in scenarios where models encounter data that differs significantly from the training set. Our approach leverages neighborhood shaping techniques to enhance the robustness of OOD detection in graph-based data. This repository contains the code and resources needed to reproduce our experiments and results.

## Installation

To get started with TopoOOD, follow these steps to install the necessary dependencies:

1. Clone the repository:

   ```bash
   git clone https://github.com/Rigelizen/TopoOOD.git
   cd TopoOOD
   ```

2. Install the required packages. You can use `pip` to install the necessary libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. If you want to access the pre-trained models and datasets, download them from the [Releases section](https://github.com/Rigelizen/TopoOOD/releases) and follow the instructions provided there.

## Usage

After installation, you can use the code in this repository to perform OOD detection on your graph data. Here’s a basic example of how to run the main script:

```bash
python main.py --dataset your_dataset --model your_model
```

Replace `your_dataset` and `your_model` with your specific dataset and model configurations. For more detailed options, run:

```bash
python main.py --help
```

This command will display all available parameters and options.

## Methodology

Our methodology combines traditional graph-based techniques with modern deep learning approaches to improve OOD detection. Key components of our approach include:

- **Neighborhood Shaping**: This technique modifies the local structure of graphs to better represent the underlying data distribution.
- **Graph Neural Networks (GNNs)**: We utilize GNNs to capture complex relationships within the data.
- **Evaluation Metrics**: We assess our model's performance using standard metrics such as AUC, F1-score, and accuracy.

### Neighborhood Shaping

Neighborhood shaping is a novel approach that involves adjusting the graph's local structure. By focusing on the relationships between nodes, we can create a more accurate representation of the data distribution. This section of the repository contains code snippets and examples demonstrating how to implement neighborhood shaping.

### Graph Neural Networks

Graph Neural Networks are essential for processing graph data. Our implementation supports various GNN architectures, allowing users to experiment with different models. Check the `models` directory for implementations of popular GNNs.

## Results

In our experiments, we benchmarked our method against several existing OOD detection techniques. The results demonstrate significant improvements in detection accuracy. Detailed results can be found in the `results` directory, which includes:

- Comparison tables
- Visualizations of model performance
- Example outputs from our method

For further insights, please refer to the original paper.

## Contributing

We welcome contributions to the TopoOOD project. If you wish to contribute, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear messages.
4. Push your branch and create a pull request.

For larger changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or inquiries, please reach out via the following methods:

- **Email**: [your.email@example.com](mailto:your.email@example.com)
- **GitHub Issues**: Feel free to open an issue in this repository.

For more resources and to download the latest releases, visit the [Releases section](https://github.com/Rigelizen/TopoOOD/releases). 

Thank you for your interest in TopoOOD! We look forward to your contributions and feedback.