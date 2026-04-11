a = ["apple", "orange", "watermelon", "strawberry", "melon", "eggplant", "flower"]


#      0         1           2            3            4        5           6

# print(a[start:end:step]) [start,end)
# Remember python is zero indexed

b = a[1 : len(a) - 1 : 1]
# ['orange', 'watermelon', 'strawberry', 'melon', 'eggplant']

c = a[1 : len(a) : 1]
# ['orange', 'watermelon', 'strawberry', 'melon', 'eggplant', 'flower']

d = a[:4]
# ['apple', 'orange', 'watermelon', 'strawberry']

e = a[2:]
# ['watermelon', 'strawberry', 'melon', 'eggplant', 'flower']

f = a[1::2]
# each step is 2 , we start at start index and then go by step and start on it
# ['orange', 'strawberry', 'eggplant']

g = a[::-1]
# Reversed Slicing , it's meaningless unless we put the step a negative number
# ['flower', 'eggplant', 'melon', 'strawberry', 'watermelon', 'orange', 'apple']

h = a[4:1:-1]
# Now because we have a reversed list the start/end indecies should be reversed
# ['melon', 'strawberry', 'watermelon']

i = a[-3:-6:-1]
# so i == h
# ['melon', 'strawberry', 'watermelon']
# print(i == h) true

# print(a)


import numpy as np

aa = np.array([[1, 2, 3, 4], [2, 2, 56, 7]])

bb = aa[:2, :3]

row_cc = aa[0, :]
list_with_row = aa[0:1:, :]


# print(row_cc, row_cc.shape)  # [ 2  2 56  7] (4,)
#
# print(list_with_row, list_with_row.shape)  # [[ 2  2 56  7]] (1, 4)

# so a num without a : is a single item slice so aa[0,1] is an num of shape () , but a[0,0:1] is a list of (1,)

ac = np.array([[1, 2], [3, 4], [5, 6]])

# [[1 2]
#  [3 4]
#  [5 6]]
# [1 4 5]


print(ac[[0, 1, 2], [0, 1, 0]])
print("hello world")
