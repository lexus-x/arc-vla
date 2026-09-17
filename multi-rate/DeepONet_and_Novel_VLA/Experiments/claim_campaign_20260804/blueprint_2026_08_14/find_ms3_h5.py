import glob, os

files = sorted(glob.glob('/home/user/maniskill_data/raw/**/*.h5', recursive=True))
for f in files:
    print(f)
