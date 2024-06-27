# Jetson config:
- Flower version: 1.9.0 
- Python 3: 3.8.0 
- Torch: 2.3.1
- Torchvision: 0.18.1
- Ubuntu: 18.04 
- virtualenv : 20.17.1
- pip: 9.0.1(old) 24.1(new)
- Setup tools: 39.0.1 (old) 68.0.0(new)
- tensorflow: 2.16.1 (pi and iMac) (2.13.1 Jetson, to be determine if it should be for all device)
- numpy: 1.26.4 (pi and iMac) 1.24.4 Jetson
- pandas: 2.2.2 (pi and iMac) (unsure for jetson, checking process stoped due to issue with installing tensorflow)
- Protobuf - 3.20.3
- Iterators - 0.0.2

# To config the Jetson:
- python3 -V (confirm below 3.8.0)
- sudo apt update
- sudo apt install python3.8
- sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
- sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
- sudo update-alternatives --config python3
- 2
- python3 -V (check if python3 is now version 3.8.0)
- python3 -m pip install --upgrade pip (make sure pip is 24.1)
- pip install flwr

- To run img_clsf,
- pip install torch torchvision
- Then cd to FL_Project and start training

- To run ids_dnp3
- for Jetson
- pip install pandas
- If you have encounter issues with "ImportError: C extension: No module named 'pandas._lib.tslib 'not built...." when running python3.server.py
- Try
- sudo pip3 uninstall pandas
- sudo pip3 install pandas
- If scikit is a issue, try
- pip install --upgrade scikit-learn
- pip install tensorflow
- 
- If HDF5 is a issue, try
- TO BE CONTINUE...

_ If Wheel is a issue, try
- python3 -m pip install --upgrade wheel
- 
- for Raspberry pi
- sudo apt-get install python3-pandas --break-system-packages
- sudo apt-get install python3-numpy==1.26.4 --break-system-packages
- sudo apt-get install python3-scikit-learn --break-system-packages
- To upgrade something, use
- python3 -m pip install --upgrade XXX --break-system-packages

# To create a Virtual Environment
- python3 -V (confirm below 3.8.0)
- sudo apt update
- sudo apt install python3.8
- sudo apt-get install python3-venv or sudo pip3 install virtualenv
- cd mkdir FL_Project
- cd FL_Project
- python3 -m venv Project_env
- ls
- source Project_env/bin/activate
- sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
- sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
- sudo update-alternatives --config python3
- 2
- python3 -V (check if python3 is now version 3.8.0)

# To install flwr 1.9.0 
- check if pip is install, if not we use “sudo apt install python3-pip”
- if “pip install flwr==0.19.0” doesn’t work and prints “No module named ‘pip’, run “python3 -m ensurepip --upgrade”
- If “ModuleNotFoundError: No module named 'pip._internal'”
- run python3 -m pip --version to see if you have pip installed.
- if yes, run python3 -m pip install --upgrade pip.
- if no, run sudo apt-get install python3-pip, and do it again
- If you have encounter issues of “ensurepip not found”, use “sudo apt-get install python3.8-venv”
- If you have encounter issues of “failed building wheel for grpcio”, try
- pip3 install --upgrade pip (from 9.0.1 to 24.0)
- python3 -m pip install --upgrade setuptools (from 39.0.1 to 68.0.0)
- pip3 install grpcio (downloaded version 1.62.2)
- After this, “pip install flwr” should work

# To find your ip address:
- In window, 
- To find your IP address, follow the steps.
- On the Windows machine, click Start. In Search, enter “command prompt” and then click Enter.
- In the command prompt, enter “ipconfig/all” and click Enter.
- Your IP address appears as an IPv4 address under Ethernet adapter Ethernet.

- In Linux/Ubuntu/Pi OS,
- To find your IP address on the command line, run:
- “ifconfig”
- To get a list of IP addresses, use the ip command. For example: 
- “ip -4 -o a | cut -d ' ' -f 2,7 | cut -d '/' -f 1”

- Use port 8080

# Trouble shoot:
- If “No module named ‘pip._internal’
- Python3 -m pip install –upgrade pip

- If “ModuleNotFoundError: No module named 'flwr'”
- pip install flwr-datasets
- python3 server.py
  
# Notes:
- To check flwr version, use “pip show flwr”

- To exit the virtual environment, use “deactivate”

- To exit the dir, use “cd”

- To access the venv in FL_Project
- cd FL_Project
- source Project_env/bin/activate

- pip install torch torchvision

- If Jetbot doesn't have enough space,
- sudo apt-get install gparted
- manually config the SD card size to it's max

# References:

https://tech.serhatteker.com/post/2019-09/upgrade-python37-on-ubuntu18/

https://www.squash.io/how-to-exit-python-virtualenv/#:~:text=The%20most%20common%20and%20recommended,restoring%20the%20system's%20default%20settings.

https://www.youtube.com/watch?v=DhLu8sI9uY4

https://flower.ai/docs/examples/embedded-devices.html#setting-up-a-jetson-xavier-nx

https://portal.perforce.com/s/article/Find-your-IP-address-and-the-port-number-for-the-server-running-Helix-Core

https://github.com/mannau/h5/blob/master/INSTALL
