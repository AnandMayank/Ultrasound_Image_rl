# Ultrasound Image Segmentation and RL Navigation

This repository contains code and documentation for a system that combines deep learning-based image segmentation with reinforcement learning to automatically navigate to regions of interest in ultrasound images.

## Project Overview

The project consists of three main components:

1. **Image Segmentation**: A ResNet-based U-Net model trained to segment regions of interest in abdominal ultrasound images.
2. **Center Detection**: An algorithm to find the centers of the segmented regions.
3. **Reinforcement Learning Navigation**: A DQN (Deep Q-Network) agent trained to navigate to the centers of the segmented regions.

## Blog Post

For a detailed explanation of the project, check out the [blog post](./abdomen_segmentation_rl_blog_post.md).

## Key Features

- ResNet-based U-Net architecture for ultrasound image segmentation
- DQN agent with experience replay for reinforcement learning navigation
- Oscillation detection and prevention mechanisms
- Momentum-based movement for smoother navigation
- Evaluation metrics and visualization tools

## Results

The trained agent successfully navigates to the centers of the segmented regions with a high success rate. The improvements to address oscillations significantly enhanced the agent's performance.

### Example Navigation

Here's an example of the agent navigating to a segmented region:

1. The agent starts at a random position in the image.
2. It uses the segmentation model to identify the region of interest.
3. It navigates to the center of the segmented region using the learned policy.
4. The agent successfully reaches the target with minimal oscillations.

## Future Work

- Training on a larger and more diverse dataset
- Implementing a continuous action space for smoother navigation
- Testing on real-time ultrasound data
- Integrating with a robotic system for physical probe positioning

## License

This project is licensed under the MIT License - see the LICENSE file for details.
