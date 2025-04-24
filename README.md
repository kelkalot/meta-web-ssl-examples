# Video-to-Image Matching & DINO Web-SSL Model Examples

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ft5Dx6vUHe0Mp2vmIJFXZI0BUHO58rId?usp=sharing) 

This repository showcases various ways to utilize and visualize features from Meta AI's new Web-ssl model series [facebook/webssl-dino300m-full2b-224](https://huggingface.co/collections/facebook/web-ssl-68094132c15fbd7808d1e9bb), Vision Transformers trained using self-supervised learning without language data. Examples range from basic feature extraction and visualization on single images to more complex applications like video analysis and content-based image retrieval (matching video frames to an image database).
The underlying Web-SSL model is the work of Meta AI (FAIR). This repository provides example code demonstrating how to use this publicly available model via the Hugging Face `transformers` library.

## Description

The code demonstrates how to leverage the rich semantic features learned by `webssl-dino300m-full2b-224`. Key examples include:

1.  **Feature Extraction:** Loading the model and extracting both global ([CLS] token) and local (patch) features.
2.  **Feature Visualization (Single Image):** Visualizing patch features using PCA and mapping the similarity of patches to the global CLS token.
3.  **Feature Visualization (Video):**
    * Generating videos showing the evolution of PCA-reduced patch features across different layers of the model for a single input image.
    * Generating videos showing the PCA-reduced *final layer* patch features for each frame of an input video.
    * Generating side-by-side videos comparing original video frames to their PCA feature visualizations.
4.  **Content-Based Matching (Video-to-Image):** Using the global [CLS] token features to find the best matching image from a custom database for each frame of an input video, and creating a side-by-side comparison video of the results.

This repository serves as a practical guide and starting point for applying this state-of-the-art vision model to various tasks.

## Features

* **Model Loading:** Demonstrates loading `facebook/webssl-dino300m-full2b-224` via `transformers`.
* **Feature Extraction:** Extracts both [CLS] token (global) and patch (local) features.
* **Single Image Visualization:**
    * PCA visualization of patch features.
    * CLS token similarity map visualization.
* **Video Feature Visualization:**
    * PCA features per-layer animation.
    * PCA features per-frame animation.
    * Side-by-side Original vs. PCA features video.
* **Video-to-Image Matching:**
    * Indexes features for a folder of database images.
    * Matches video frames to database images using cosine similarity on [CLS] features.
    * Applies a similarity threshold to filter matches.
    * Generates a side-by-side Video Frame vs. Best Match video.

## Demo / Results

The code includes several visualization examples. The GIF below shows the PCA visualization of patch features evolving across the different layers of the transformer for a single input image. The final example in the code focuses on the video-to-image matching application, generating a side-by-side comparison video (examples in the `/results` folder).

![Demo GIF: PCA Features Across Layers](https://github.com/kelkalot/meta-web-ssl-examples/blob/main/results/image_layers_pca_animation_webssl-dino300m-full2b-224.gif)

*(You can find other example output videos, including the results from the video-to-image matching, in the `/results` folder of this repository.)*

## Repository Structure

*Note: The `.py` and `.ipynb` files contain multiple distinct examples as described above. You may want to run specific sections or adapt them.*

## Usage

This repository contains several examples. The primary focus for generating the matching video is the **final example** in the script/notebook.

### For Video-to-Image Matching Example:

1.  **Prepare Data:**
    * Create folders for your data (e.g., using the `data/` structure shown above).
    * Place your input video file in a specific location (e.g., `data/video/your_video.mp4`).
    * Place all the images for your matching database inside a separate folder (e.g., `data/images/`).

2.  **Configure Paths and Parameters:**
    Open `web_ssl_model_examples.py` or `web_ssl_model_examples.ipynb`. Locate the **final example section** (likely marked with comments about matching images to video frames) and modify the configuration variables:
    * `video_path`: Set the full path to your input video file.
    * `image_folder_path`: Set the full path to the folder containing your database images.
    * `output_dir`: Specify where the output video should be saved.
    * `model_name`: Ensure `'facebook/webssl-dino300m-full2b-224'` is selected (or change if testing others).
    * `match_score_threshold`: Adjust this (0.0 to 1.0) to control matching strictness. Higher values filter more. Recommended starting point: 0.6 - 0.7.
    * `max_frames_to_process`: Set to `None` for the whole video or an integer for testing.
    * `panel_size`, `output_fps`: Adjust output video aesthetics if needed.

3.  **Run the Code:**
    * **Python Script:** Execute the script. Note that it might run *all* examples unless you comment out or modify the script to run only the final section.
        ```bash
        python web_ssl_model_examples.py
        ```
    * **Jupyter Notebook:** Open `web_ssl_model_examples.ipynb` and run the cells corresponding to the video-to-image matching example (likely the last major section).
    * **Google Colab:** Use the "Open In Colab" badge. Upload data or mount Drive, update paths in the relevant notebook section, and run those cells.

### Other Examples

The script/notebook also contains self-contained examples for:
* Basic feature extraction.
* PCA and CLS similarity visualization for single images.
* PCA visualization of features across model layers (creating a video).
* PCA visualization of final layer features per video frame (creating a video).
* Side-by-side video of original vs. PCA features.

You can run these sections independently by adapting the code or running the relevant notebook cells.

## Model Information

### Model Showcased

This repository primarily demonstrates the use of **Web-SSL DINO ViT-300M** (`facebook/webssl-dino300m-full2b-224`).

### Links

* **Model used for this example on Hugging Face:** [facebook/webssl-dino300m-full2b-224](https://huggingface.co/facebook/webssl-dino300m-full2b-224)
* **Web-SSL Collection:** [facebook/web-ssl](https://huggingface.co/collections/facebook/web-ssl-68094132c15fbd7808d1e9bb)

### Details

* **Description:** Web-SSL DINO 300M is a 300 million parameter Vision Transformer model trained using self-supervised learning on 2 billion web images without language supervision. This model demonstrates that pure visual learning, when scaled appropriately, can match or exceed the performance of language-supervised models like CLIP across various vision tasks.
* **Architecture:** ViT (1536 width, 40 depth, 24 heads)
* **Parameters:** 300M
* **Resolution:** 224×224 pixels
* **Training:** Self-supervised Web-DINO on 2B image samples from MetaCLIP web data.
* **Paper:** "Scaling Language-Free Visual Representation Learning" (Fan et al., 2025).

Please cite the original paper of the authors if you use it for your work:

 ```bibtex
@article{fan2025scaling,
  title={Scaling Language-Free Visual Representation Learning},
  author={David Fan and Shengbang Tong and Jiachen Zhu and Koustuv Sinha and Zhuang Liu and Xinlei Chen and Michael Rabbat and Nicolas Ballas and Yann LeCun and Amir Bar and Saining Xie},
  year={2025},
  eprint={2504.01017},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

