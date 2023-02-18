#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 20:39:31 2022

@author: jrdonoso
"""
import ADS1256
import RPi.GPIO as GPIO
import socket
import sys
from time import sleep
import random
from struct import pack
from datetime import datetime

try:
    ADC = ADS1256.ADS1256()
    ADC.ADS1256_init()

except :
    GPIO.cleanup()
    print ("\r\nProgram end     ")
    exit()

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host, port = '0.0.0.0', 5005
sock.bind((host,port))
sock.listen(2)

while True:
    clientsocket, address = sock.accept()
    print(f'Connection from {address} has been established...')
    ADC_Value = ADC.ADS1256_GetAll()
    ch0 = ADC_Value[0]*5.0/0x7fffff # ADC.ADS1256_GetChannalValue(0)*5.0/0x7fffff
    ch1 = ADC_Value[1]*1.0/0x7fffff #ADC.ADS1256_GetChannalValue(1)*1.0/0x7fffff
    ch2 = ADC_Value[2]*5.0/0x7fffff #ADC.ADS1256_GetChannalValue(2)*5.0/0x7fffff
    r_test = ch0
    lumi = 0.9 - ch1
    g_skin = 1.0/ch2 #nS
    if g_skin<0.32:
        g_skin = 0.0
    # Pack three 32-bit floats into message and send
    message = pack('3f', r_test, lumi, g_skin)
    #ch0, ch1, ch2 = round(r_,3), round(ch1,3), round(ch2,3)
    try:
        #sock.sendto(message, server_address)
        clientsocket.send(message)
        print(datetime.now().strftime('%H:%M:%S.%f')[:-4] + f'--> R_test: {round(r_test,2)}, Lumi: {round(lumi,3)}, g_skin: {round(g_skin,3)}')
        clientsocket.close()
    except OSError:
        print("Network Error")