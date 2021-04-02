# Video-Player-Controlled-by-Action-Recognition

 在src路径下运行python realTimeReading.py输出识别结果，运行python ../serialTest/serial_test.py输出串口数据

<br/>

<br/>

<br/>

#### Data Collection

数据(csv)格式

| Column 1              | Column 2              | Column 3              | Label       |
| --------------------- | --------------------- | --------------------- | ----------- |
| Channel 1（如图）数据 | Channel 2（如图）数据 | Channel 3（如图）数据 | 0/1/2/3/4/5 |

<br/>

注：Label:

| 动作           | Label |
| -------------- | ----- |
| Left Rotation  | 0     |
| Right Rotation | 1     |
| Left Waving    | 2     |
| Right Waving   | 3     |
| Fist/Snap      | 4     |

<br/>

<br/>

EMG电极位置

![image](https://github.com/Diregie-J/Video-Player-Controlled-by-Action-Recognition/blob/main/IMG/channel_places.png)
