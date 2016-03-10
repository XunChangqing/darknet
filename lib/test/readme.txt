1. 文件目录说明
lib\                          库文件目录
  libobject_detection.h       头文件
  libobject_detection.so      库文件
model\                        模型文件目录
  448_version_1.cfg           模型配置文件
  448_version_1.model         模型参数文件
test_images\                  测试图像目录
  person.jpg                  测试输入图像文件
test.c                        测试程序源文件
Makefile                      测试程序Makefile
readme.txt                    说明

2. 系统需求
UBUNTU 1404 LTS 64bit系统
带有NVIDIA GTX 960以上显卡
安装Nvidia CUDA SDK 7.5

3. 运行说明
库程序本身不依赖于OpenCV，但是测试程序的编译依赖于OpenCV，可以通过如下命令安装OpenCV：
sudo apt-get install

a. 编译test程序
make

b. 执行测试程序
./test
//如果报无法找到libobject_detction.so文件，则执行一下命令
export LD_LIBRARY_PATH=./lib

4. 库说明
最好输入D1分辨率的图片，过大的分辨率会导致较多的漏检
