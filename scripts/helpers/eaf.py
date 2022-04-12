import os
import pympi.Elan as Elan

def get_eaf_file(wav_file):

    eaf_path   = os.path.splitext(wav_file)[0] + ".eaf"
    eaf_exists = os.path.exists(eaf_path)

    return eaf_path, eaf_exists
