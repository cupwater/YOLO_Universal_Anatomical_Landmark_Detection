'''
Author: Peng Bo
Date: 2022-05-28 23:48:22
LastEditTime: 2022-05-29 20:19:49
Description: 

'''
from .cephalometric import Cephalometric
from .hand import Hand
from .chest import Chest
from .chest_inmemory import ChestInmemory
from .chest_test import ChestTest
from .chest_26lms import ChestInmemory26LMS

def get_dataset(s):
    return {
            'cephalometric':Cephalometric,
            'hand':Hand,
            'chest_26':ChestInmemory26LMS,
            'chest_6':ChestInmemory,
            'chest':Chest,
            'chest_test':ChestTest,
           }[s.lower()]


