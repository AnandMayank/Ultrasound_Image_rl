---
layout: default
---

# Ultrasound Image Segmentation and RL Navigation

## Project Overview

This project combines deep learning-based image segmentation with reinforcement learning to automatically navigate to regions of interest in ultrasound images. The system consists of three main components:

1. **Image Segmentation**: A ResNet-based U-Net model trained to segment regions of interest in abdominal ultrasound images.
2. **Center Detection**: An algorithm to find the centers of the segmented regions.
3. **Reinforcement Learning Navigation**: A DQN (Deep Q-Network) agent trained to navigate to the centers of the segmented regions.

## Blog Posts

- [Ultrasound Image Segmentation and Reinforcement Learning Navigation](./abdomen_segmentation_rl_blog_post.md)

## Results

The trained agent successfully navigates to the centers of the segmented regions with a high success rate. The improvements to address oscillations significantly enhanced the agent's performance.

### Example Navigation

Here's an example of the agent navigating to a segmented region:

1. The agent starts at a random position in the image.
2. It uses the segmentation model to identify the region of interest.
3. It navigates to the center of the segmented region using the learned policy.
4. The agent successfully reaches the target with minimal oscillations.

## Repository

The complete code for this project is available on [GitHub](https://github.com/AnandMayank/ultrasound-segmentation-rl).
