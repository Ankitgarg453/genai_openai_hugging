Follow the below steps to setup your webapp on AWS EC2. 
Make aure to setup your inbound connections in security groups after below steps.

first login to the AWS: https://aws.amazon.com/console/

search about the EC2

you need to config the UBUNTU Machine

launch the instance

update the machine:

sudo apt update

sudo apt-get update

sudo apt upgrade -y

sudo apt install git curl unzip tar make sudo vim wget -y

sudo apt install python3-pip

sudo apt update && sudo apt install python3-venv -y

git clone "Your-repository"

python3 -m venv myenv

pip3 install -r requirements.txt

python3 -m streamlit run StreamlitAPP.py  -- if runs on this

if you want to add openai api key
create .env file in your server touch .env
vi .env #press insert #copy your api key and paste it there #press and then :wq and hit enter

go with security and add the inbound rule add the port 8501
