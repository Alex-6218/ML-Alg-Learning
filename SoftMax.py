import numpy as np
import matplotlib.pyplot as plt
import sys, os

train_files_path = ''
test_file = ''
match sys.argv[1]:
    case 'train':
        print(f"Training mode selected.")
        train_files_path = input("Provide the path of the training file:")  
        for entry in os.scandir(f'{train_files_path}'):
            if entry.is_file() and (entry.name.endswith('.jpg') or entry.name.endswith('.png')):
                print(f"Processing file: {entry.name}")
                plt.imread(f'{train_files_path}/{entry.name}')
    case 'test':
        print(f"Testing mode selected.")
        test_file_path = f'./{input("Provide the path of the test file:")}'
    case _:
        print(f"Invalid argument. Please use 'train' or 'test'.")
        sys.exit(1)

