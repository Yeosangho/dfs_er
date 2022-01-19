
pip3 install --user numpy
pip3 install --user moviepy
#pip3 install --user matplot
pip3 install --user scipy
pip3 install --user opencv-python
git clone https://github.com/openai/gym.git
git clone https://github.com/openai/baselines.git
cd gym
pip3 install --user gym
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
pip3 install --user -e '.[atari]'

