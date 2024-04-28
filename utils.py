import os
import time

def runtime():
    timestamp = time.time()  
    local_time = time.localtime(timestamp)  
    year, month, day, hour, minute, second, weekday, yearday, isdst = local_time    
    result = f"{year}_{month}_{day}_{hour}_{minute}_{second}"
    return result

def mkdir(path):
    name = runtime() 
    path = f'{path}/{name}'
    os.popen(f'mkdir -p {path}/src')
    os.popen(f'cp ./*.py {path}/src')
    return path

