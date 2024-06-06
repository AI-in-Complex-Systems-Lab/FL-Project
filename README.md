# Version list:
Flower version: 0.19.0 (1.8.0 max)  
Python 3: 3.7.0 and above (3.12 max)  
Ubuntu: 18.04   
virtualenv : 20.17.1  
pip - 9.0.1(old) 24.0(new)  
Setup tools - 39.0.1 (old) 68.0.0(new)  
numpy -1.20.3  
Protobuf - 3.20.3  
Iterators - 0.0.2  

# To create a Virtual Environment
python3 -V (confirm below 3.7.0)  
sudo apt update  
sudo apt install python3.7  
sudo apt-get install python3-venv or sudo pip3 install virtualenv  
cd mkdir FL_Project  
cd FL_Project  
python3 -m venv Project_env  
ls  
source Project_env/bin/activate  
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1  
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2  
sudo update-alternatives --config python3  
2  
python3 -V (check if python3 is now version 3.7.5)  

# To install flwr 0.19.0 in the venv  
check if pip is install, if not we use “sudo apt install python3-pip”  
if “pip install flwr==0.19.0” doesn’t work and prints “No module named ‘pip’, run “python3 -m ensurepip --upgrade”  
If you have encounter issues of “ensurepip not found”, use “sudo apt-get install python3.7-venv”  
If you have encounter issues of “failed building wheel for grpcio”  
Try  
pip3 install --upgrade pip (from 9.0.1 to 24.0)  
python3 -m pip install --upgrade setuptools (from 39.0.1 to 68.0.0)  
pip3 install grpcio (downloaded version 1.62.2)  
After this, “pip install flwr==0.19.0” should work  
  
## Notes:
To check flwr version, use “pip show flwr”  

To exit the virtual environment, use “deactivate”  

To exit the dir, use “cd”  

To access the venv in FL_Project  
cd FL_Project  
source Project_env/bin/activate  


## References:  

https://tech.serhatteker.com/post/2019-09/upgrade-python37-on-ubuntu18/

https://www.squash.io/how-to-exit-python-virtualenv/#:~:text=The%20most%20common%20and%20recommended,restoring%20the%20system's%20default%20settings.

https://www.youtube.com/watch?v=DhLu8sI9uY4








