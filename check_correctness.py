import numpy as np

for network_file in [
    "grid_150_150.txt",
    "random_20000_40000.txt",
    "dense_1000.txt",
]:
    my_output = np.loadtxt(
        f"22210690089_output_{network_file.split('_')[0]}.txt",
        skiprows=1,
        delimiter=",",
        dtype=np.int32,
    )
    correct_output = np.loadtxt(
        f"答案_Output_{network_file}",
        skiprows=1,
        delimiter=",",
        dtype=np.int32,
    )
    print("错误的个数是：", sum((my_output != correct_output)))
