import json, sys, os
OUT_PATH = r'g:\\Master\\Thesis\\FLT\\Code\\ECG-to-stress\\notebooks\\_test_out.txt'
try:
    msg = 'hello world'
    with open(OUT_PATH, 'w') as f:
        f.write(msg)
except Exception as e:
    with open(OUT_PATH, 'w') as f:
        f.write('ERROR: ' + str(e))


