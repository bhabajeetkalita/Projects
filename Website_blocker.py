'''
Author: Bhabajeet Kalita
Description : A program to block certain websites for a particular time from 8am to 2pm everyday
'''
import time
from datetime import datetime as dt
hosts_temp="/Users/Gourhari/Desktop/hosts"
hosts_path = "/private/etc/hosts"
redirect="127.0.0.1"
website_list=["www.facebook.com","facebook.com","wikipedia.com","gmail.com"]
while True:
    if dt(dt.now().year,dt.now().month,dt.now().day,8)<dt.now()<dt(dt.now().year,dt.now().month,dt.now().day,14):
        with open(hosts_path,'r+') as file:
            content=file.read()
            for website in website_list:
                if website in content:
                    pass
                else:
                    file.write(redirect+" "+website+"\n")
    else:
        with open(hosts_path,'r+') as file:
            content = file.readlines()
            file.seek(0)
            for line in content:
                if not any(website in line for website in website_list):
                    file.write(line)
            file.truncate()
        print("Fun hours...")
    time.sleep(5)
