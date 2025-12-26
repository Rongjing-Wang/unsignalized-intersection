### Environment Setup
```
conda create -n DRL python=3.6
conda activate DRL
pip install numpy==1.16.2
pip install opencv-contrib-python==3.4.2.16
pip install opencv-python==4.2.0.32
pip install tensorflow==1.12.0
pip install matplotlib==3.0.2
pip install scipy==1.2.1
pip install pandas==0.24.2
pip install pillow==6.2.2
```

### Train a model:
```
python main.py --mat_path arvTimeNewVeh_for_train.mat --type train --exp_name train_demo --num_episodes 1

```

### Test the model:
```
python main.py --exp_name train_demo --mat_path arvTimeNewVeh_new_1000_12.mat  --type test  --visible --video_name test
```
Note:the visual prarameters "--visible" and "--video_name" is optional. If use the "--visible", there will be a simulation interface to show the running interface of the vehicle in real time. the "--video_name test" is used to generate a video ,named "test.avi", saved in "./result_imgs/".

### Batch testing
```
python main.py --exp_name train_demo --batch_test --visible --video_name test
```
