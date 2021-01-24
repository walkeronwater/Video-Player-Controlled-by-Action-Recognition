import serial
import time
import sys
from time import sleep
import vis

# change filename, port number, and baudrate if needed
filename = 'temp'+'33'+'.csv'
port = "COM9"
baudrate = 19200

ser = serial.Serial(port, baudrate)
ser.set_buffer_size(rx_size = 2147483647, tx_size = 2147483647)

sleep(2)

ser.write("a".encode())
print("Start")
sleep(0.01)

f = open(filename, 'w')

i=0
ser.flushInput()
ser.flushOutput()
for i in range(300):
    #time1 = time.time()
    f.write(str(ser.readline().strip().decode('utf-8')))
    f.write(',')
    f.write(str(ser.readline().strip().decode('utf-8')))
    f.write(',')
    f.write(str(ser.readline().strip().decode('utf-8')))
    f.write(',')
    f.write('0') # label
    f.write('\n')


    f.close()
    f = open(filename, 'a')
    #time2=time.time()-time1
    #print(time2)

vis.visFile(filename)