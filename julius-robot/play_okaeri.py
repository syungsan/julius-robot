#! /usr/bin/env python

# This script is for windows only.
# Please fix if you want to use it on Linux.

import os
import winsound

OKAERI_FILE = './voice-okaeri-yasashime.wav'

def play():
  winsound.PlaySound(OKAERI_FILE, winsound.SND_FILENAME)

if __name__ == '__main__':
  play()

