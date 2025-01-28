import serial
import time

ser = serial.Serial('/dev/ttyS0',9600,timeout=1)
ser.reset_input_buffer()

try:
    while True:
        time.sleep(1)
        ser.write("hi".encode('utf-8'))

except KeyboardInterrupt:
    ser.close()