import os
import sys
import traceback

all_tests = [
    'dataset generation test',
    'sparse coding test 1',
    'sparse coding test 2',
    'sparse coding test 3',
    'sparse coding test 4',
    'ista_fista test 1',
    'ista_fista test 2',
    'ista_fista test 3',
    'sparse coding test 5',
    ]

# terminal colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_traceback():
  print('---- Here is the traceback ----')
  _, _, tb = sys.exc_info()
  traceback.print_tb(tb) # Fixed format
  exit(1)

def print_failure(test_name):
  print(f'{bcolors.FAIL}  {test_name.capitalize()} failed{bcolors.ENDC}')

def print_success(test_name):
  print(f'{bcolors.OKGREEN}  {test_name.capitalize()} succeeded{bcolors.ENDC}')


def main():
  print('Running tests on VTC library...')

  print(f'{bcolors.WARNING}  Running {all_tests[0]} {bcolors.ENDC}')
  try:
    import dset_generation_1
  except:
    print_failure(all_tests[0])
    print_traceback()
  print_success(all_tests[0])

  print(f'{bcolors.WARNING}  Running {all_tests[1]} {bcolors.ENDC}')
  try:
    import sparse_coding_1
  except:
    print_failure(all_tests[1])
    print_traceback()
  print_success(all_tests[1])

  print(f'{bcolors.WARNING}  Running {all_tests[2]} {bcolors.ENDC}')
  try:
    import sparse_coding_2
  except:
    print_failure(all_tests[2])
    print_traceback()
  print_success(all_tests[2])

  print(f'{bcolors.WARNING}  Running {all_tests[3]} {bcolors.ENDC}')
  try:
    import sparse_coding_3
  except:
    print_failure(all_tests[3])
    print_traceback()
  print_success(all_tests[3])

  print(f'{bcolors.WARNING}  Running {all_tests[4]} {bcolors.ENDC}')
  try:
    import sparse_coding_4
  except:
    print_failure(all_tests[4])
    print_traceback()
  print_success(all_tests[4])

  print(f'{bcolors.WARNING}  Running {all_tests[5]} {bcolors.ENDC}')
  try:
    import ista_fista_1
  except:
    print_failure(all_tests[5])
    print_traceback()
  print_success(all_tests[5])

  print(f'{bcolors.WARNING}  Running {all_tests[6]} {bcolors.ENDC}')
  try:
    import ista_fista_2
  except:
    print_failure(all_tests[6])
    print_traceback()
  print_success(all_tests[6])

  print(f'{bcolors.WARNING}  Running {all_tests[7]} {bcolors.ENDC}')
  try:
    import ista_fista_3
  except:
    print_failure(all_tests[7])
    print_traceback()
  print_success(all_tests[7])

  print(f'{bcolors.WARNING}  Running {all_tests[8]} {bcolors.ENDC}')
  try:
    import sparse_coding_5
  except:
    print_failure(all_tests[8])
    print_traceback()
  print_success(all_tests[8])

  print(f'{bcolors.OKGREEN}---------------------------------')
  print(f'All tests completed successfully')
  print(f'-----------------------------------{bcolors.ENDC}')

if __name__ == '__main__':
  main()
