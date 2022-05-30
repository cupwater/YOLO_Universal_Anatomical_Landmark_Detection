from .cephalometric import Cephalometric
from .hand import Hand
from .chest import Chest
from .chest_inmemory import ChestInmemory
from .chest_26lms import ChestInmemory26LMS

def get_dataset(s):
    return {
            'cephalometric':Cephalometric,
            'hand':Hand,
            'chest_26':ChestInmemory26LMS,
            'chest_6':ChestInmemory,
            'chest':Chest,
           }[s.lower()]


