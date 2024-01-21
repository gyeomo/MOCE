# MOCE

What Does a Model Really Look at?: Extracting Model-Oriented Concepts for Explaining Deep Neural Networks

Accepted in IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

# Requirements
CUDA == 11.1
cudnn == 8.1.0
matplotlib == 3.3.4
numpy == 1.19.5
opencv-python == 4.5.3.56
pillow == 8.3.1
scikit-learn == 0.24.2
scikit-image == 0.17.2
scipy == 1.5.4
torch == 1.7.1+cu110
torchvision == 0.8.2+cu110
tornado == 6.1

# Implementation
Set the dataset in the "./dataset" folder
Since we are using the ImageNet label, the folder in "./dataset" should be shaped like an 'n02123045'.
Set the target model and layer number in main.py.
The layer number consists of 0~15, and it consists of the conv layer and the bottleneck layer of each model.

When setup is complete, run with this:

python main.py

# Acknowledgement
@inproceedings{ghorbani2019towards,
  title={Towards automatic concept-based explanations},
  author={Ghorbani, Amirata and Wexler, James and Zou, James Y and Kim, Been},
  booktitle={Advances in Neural Information Processing Systems},
  pages={9273--9282},
  year={2019}
}

@inproceedings{netdissect2017,
  title={Network Dissection: Quantifying Interpretability of Deep Visual Representations},
  author={Bau, David and Zhou, Bolei and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
  booktitle={Computer Vision and Pattern Recognition},
  year={2017}
}
